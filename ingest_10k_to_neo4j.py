#!/usr/bin/env python3

"""
ingest_10k_to_neo4j.py

Ingest inline-XBRL (JSON) and primary HTML/TXT from 10-K filing(s) into a Neo4j knowledge graph.

- Input: 
    Single folder mode: a folder containing files like:
        - *_meta.json         (high-level filing metadata produced by your my_sec_api_test.py)
        - *_10k.json          (XBRL JSON or a JSON bundle with `xbrl_data`)
        - *_primary.html      (primary iXBRL HTML)
        - *_primary.txt       (plain-text extracted body)
    
    Recursive mode: a parent folder containing multiple subfolders, each with the above files
        (e.g., 10k/COMPANY_NAME/2020/, 10k/COMPANY_NAME/2021/, etc.)

- Output: nodes & relationships in a local Neo4j database:
    Organization, Filing, Fact, Concept, Unit, Period, Context, Dimension,
    Security, Exchange, Section, Sentence (+ optional Topic, if requested).

Run:
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=neo4j_password
    
    # Single folder
    python ingest_10k_to_neo4j.py /path/to/single/folder [--extract-topics]
    
    # Multiple folders (recursive)
    python ingest_10k_to_neo4j.py /path/to/parent/folder --recursive [--extract-topics]

This script is idempotent: it uses MERGE for stable identifiers.
"""
import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from bs4 import BeautifulSoup  # type: ignore
from neo4j import GraphDatabase  # type: ignore

# ------------------------------
# Data classes & utilities
# ------------------------------

@dataclass
class Period:
    start: Optional[str] = None
    end: Optional[str] = None
    instant: Optional[str] = None
    
    @property
    def id(self) -> str:
        """Generate a stable ID for this period based on its properties."""
        return stable_hash(
            self.start or "",
            self.end or "",
            self.instant or ""
        )

@dataclass
class Dimension:
    axis: str
    member: str

@dataclass
class FactRecord:
    id: str
    concept: str
    value: Union[str, float, int, bool, None]
    value_type: str
    unit: Optional[str]
    period: Period
    context_ref: Optional[str]
    decimals: Optional[str]
    dimensions: List[Dimension]
    source_path: str

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to was were will with without within into over under between during including until while than then so such via""".split())

def stable_hash(*parts: str, maxlen: int = 40) -> str:
    h = hashlib.md5("|".join([p if p is not None else "" for p in parts]).encode("utf-8")).hexdigest()
    return h[:maxlen]

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def guess_files(folder: Path) -> Dict[str, Optional[Path]]:
    files = { "meta": None, "xbrl": None, "html": None, "txt": None }
    for p in folder.glob("*"):
        name = p.name.lower()
        if name.endswith("_meta.json"):
            files["meta"] = p
        elif name.endswith("_10k.json"):
            files["xbrl"] = p
        elif name.endswith("_primary.html"):
            files["html"] = p
        elif name.endswith("_primary.txt"):
            files["txt"] = p
    return files

# ------------------------------
# Neo4j schema
# ------------------------------

SCHEMA_QUERIES = [
    # Organizations & filings
    "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.cik IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Filing) REQUIRE f.accession IS UNIQUE",
    # Facts & supporting nodes
    "CREATE CONSTRAINT IF NOT EXISTS FOR (fact:Fact) REQUIRE fact.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (u:Unit) REQUIRE u.measure IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Period) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (ctx:Context) REQUIRE ctx.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dimension) REQUIRE (d.axis, d.member) IS NODE KEY",
    # Securities & exchanges
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Security) REQUIRE (s.title) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (ex:Exchange) REQUIRE ex.code IS UNIQUE",
    # Sections & sentences
    "CREATE CONSTRAINT IF NOT EXISTS FOR (sec:Section) REQUIRE (sec.filingAccession, sec.item, sec.title) IS NODE KEY",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (sent:Sentence) REQUIRE sent.id IS UNIQUE",
    # Topics (optional)
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    # Full-text index on Sentence.text for narrative search
    "CREATE FULLTEXT INDEX sentenceText IF NOT EXISTS FOR (s:Sentence) ON EACH [s.text]",
]

def apply_schema(driver):
    with driver.session() as session:
        for q in SCHEMA_QUERIES:
            session.run(q)

# ------------------------------
# XBRL helpers
# ------------------------------

def dict_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def is_fact_like(node: Any) -> bool:
    """
    Heuristic to identify a fact-y object emitted by your pipeline.
    We treat a dict as a 'fact' if it has a 'value' and either a 'period', 'contextRef', or 'segment'/'unit' hint.
    """
    if not isinstance(node, dict):
        return False
    if "value" not in node:
        return False
    if any(k in node for k in ("period", "context", "contextRef", "segment", "unit", "decimals")):
        return True
    # Allow scalar facts (e.g., strings on the cover page) with just 'value'
    return True

def value_type_of(v: Any) -> str:
    if isinstance(v, bool): return "boolean"
    if isinstance(v, int): return "integer"
    if isinstance(v, float): return "float"
    if v is None: return "null"
    return "string"

def to_period(period_obj: Any) -> Period:
    if isinstance(period_obj, dict):
        return Period(period_obj.get("startDate"), period_obj.get("endDate"), period_obj.get("instant"))
    return Period()

def to_dimensions(segment_list: Any) -> List[Dimension]:
    dims: List[Dimension] = []
    if isinstance(segment_list, list):
        for seg in segment_list:
            axis = dict_get(seg, "dimension")
            member = dict_get(seg, "value")
            if axis and member:
                dims.append(Dimension(axis=axis, member=str(member)))
    return dims

def extract_facts_from_json(xbrl_json: Any, concept_path: str = "") -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Recursively walk json, yielding (concept_path, fact_obj) pairs.
    concept_path is a slash-joined hierarchy of keys that led to the fact.
    """
    if isinstance(xbrl_json, dict):
        # prefer common shapes first
        if is_fact_like(xbrl_json):
            yield (concept_path.strip("/"), xbrl_json)
        for k, v in xbrl_json.items():
            sub_path = f"{concept_path}/{k}" if concept_path else k
            yield from extract_facts_from_json(v, sub_path)
    elif isinstance(xbrl_json, list):
        for idx, item in enumerate(xbrl_json):
            sub_path = f"{concept_path}[{idx}]"
            yield from extract_facts_from_json(item, sub_path)

def normalize_concept(path: str) -> str:
    """
    Collapse array indices and keep the last named component as the 'concept'.
    e.g., 'xbrl_data/CoverPage/TradingSymbol[2]' -> 'CoverPage.TradingSymbol'
    """
    parts = [p for p in re.split(r"[\/]", path) if p]
    parts = [re.sub(r"\[\d+\]", "", p) for p in parts]
    if not parts:
        return "UnknownConcept"
    # Truncate leading 'xbrl_data' if present
    if parts[0] == "xbrl_data":
        parts = parts[1:]
    # Keep at most last 2 levels for readability
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return parts[-1]

def build_fact_records(xbrl_json: Any, cik: str, accession: str, source_path: str) -> List[FactRecord]:
    facts: List[FactRecord] = []
    for path, obj in extract_facts_from_json(xbrl_json):
        try:
            if not is_fact_like(obj):
                continue
            concept = normalize_concept(path)
            val = obj.get("value")
            unit = obj.get("unit") or obj.get("unitRef") or None
            decimals = obj.get("decimals")
            context_ref = obj.get("contextRef") or dict_get(obj, "context", {}).get("id")
            period = to_period(obj.get("period"))
            dims = to_dimensions(obj.get("segment") or [])
            # Id hash includes concept, period, dims, and value
            dims_key = ";".join([f"{d.axis}={d.member}" for d in dims]) if dims else ""
            pid = stable_hash(cik, accession, concept, period.id, dims_key, str(val))
            facts.append(FactRecord(
                id=pid,
                concept=concept,
                value=val,
                value_type=value_type_of(val),
                unit=unit,
                period=period,
                context_ref=context_ref,
                decimals=str(decimals) if decimals is not None else None,
                dimensions=dims,
                source_path=str(source_path),
            ))
        except Exception as e:
            # Skip malformed facts but keep going
            continue
    return facts

# ------------------------------
# Narrative parsing
# ------------------------------

ITEM_RE = re.compile(r"^\s*item\s+(\d+[A-Z]?)\.\s*(.+?)\s*$", re.IGNORECASE)

def extract_sections_and_sentences_from_html(html_text: str) -> List[Tuple[str, str, List[str]]]:
    """
    Returns list of (item, title, sentences) tuples. Item may be "1", "1A", etc.
    We look for headings (h1..h4) and also raw text lines starting with "Item X."
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # Gather candidate headings
    candidates: List[Tuple[str, str, str]] = []  # (item, title, raw_block_text)
    # 1) Headings
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "b", "strong"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        m = ITEM_RE.match(text)
        if m:
            candidates.append((m.group(1), m.group(2), ""))

    # 2) Text lines with "Item N."
    lines = [l.strip() for l in soup.get_text("\n").split("\n")]
    for line in lines:
        m = ITEM_RE.match(line)
        if m:
            candidates.append((m.group(1), m.group(2), ""))

    # Deduplicate preserving order
    seen = set()
    uniq: List[Tuple[str, str, str]] = []
    for c in candidates:
        key = (c[0].upper(), c[1].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    # Split into sections by item markers in the plain text
    sections: List[Tuple[str, str, List[str]]] = []
    # Build a regex with all found items to split on
    if not uniq:
        # fallback: one giant section
        big_text = soup.get_text("\n", strip=True)
        sentences = split_into_sentences(big_text)
        return [("ALL", "Document", sentences)]

    items_order = [i for i, _, _ in uniq]
    pattern = re.compile(rf"(?i)\b(item\s+(?:{'|'.join([re.escape(i) for i in items_order])})\.)")

    text = soup.get_text("\n", strip=True)
    chunks = pattern.split(text)

    current_item = "ALL"
    current_title = "Document"
    buffer = []
    for chunk in chunks:
        m = ITEM_RE.match(chunk)
        if m:
            # flush
            if buffer:
                sentences = split_into_sentences("\n".join(buffer))
                sections.append((current_item, current_title, sentences))
                buffer = []
            current_item = m.group(1).upper()
            current_title = m.group(2).strip()
        else:
            buffer.append(chunk)
    if buffer:
        sentences = split_into_sentences("\n".join(buffer))
        sections.append((current_item, current_title, sentences))

    # Post-process: clean and limit sentence size
    clean_sections: List[Tuple[str, str, List[str]]] = []
    for item, title, sents in sections:
        sents = [s.strip() for s in sents if 10 <= len(s.strip()) <= 1200]
        if sents:
            clean_sections.append((item, title, sents))
    return clean_sections

def split_into_sentences(text: str) -> List[str]:
    # simple conservative splitter
    # normalize spaces
    text = re.sub(r"\s+", " ", text)
    # split on period/question/exclamation followed by space & capital
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z\(\[])",
                     text)
    return [p.strip() for p in parts if p and not p.isspace()]

def extract_topics(sentences: List[str], top_k: int = 40) -> List[str]:
    """
    Very simple keyphrase extraction: take frequent unigrams/bigrams excluding stopwords and numeric-only tokens.
    """
    counts = {}
    def add(token: str):
        token = token.lower()
        if not token or token in STOPWORDS:
            return
        if re.fullmatch(r"[0-9\.\,\-\%]+", token):
            return
        if len(token) <= 2:
            return
        counts[token] = counts.get(token, 0) + 1

    for s in sentences:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", s)]
        for w in words:
            add(w)
        # basic bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i].lower()} {words[i+1].lower()}"
            if all(w not in STOPWORDS for w in bigram.split()):
                counts[bigram] = counts.get(bigram, 0) + 1

    # top_k
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items[:top_k]]

# ------------------------------
# Ingestion
# ------------------------------

def find_10k_folders(root_folder: Path) -> List[Path]:
    """
    Recursively find all folders that contain 10-K files.
    Returns a list of paths to folders containing the expected file patterns.
    """
    folders_with_10k = []
    
    def has_10k_files(folder: Path) -> bool:
        """Check if a folder contains the expected 10-K file patterns."""
        files = guess_files(folder)
        # A folder is considered to have 10-K files if it has at least meta and one other file type
        return files["meta"] is not None and any(files[k] is not None for k in ["xbrl", "html", "txt"])
    
    # Check if the root folder itself contains 10-K files
    if has_10k_files(root_folder):
        folders_with_10k.append(root_folder)
    
    # Recursively check subdirectories
    try:
        for item in root_folder.iterdir():
            if item.is_dir():
                if has_10k_files(item):
                    folders_with_10k.append(item)
                else:
                    # Recursively check deeper levels
                    folders_with_10k.extend(find_10k_folders(item))
    except PermissionError:
        print(f"Warning: Permission denied accessing {root_folder}")
    
    return sorted(folders_with_10k)

def ingest_folder(folder: Path, extract_topics_flag: bool = False):
    files = guess_files(folder)
    if not any(files.values()):
        raise FileNotFoundError(f"No expected files found in {folder}")

    meta = {}
    if files["meta"]:
        meta = load_json(files["meta"])

    # pull basics from meta if available
    company_name = meta.get("company") or meta.get("EntityRegistrantName") or "Unknown Company"
    cik = meta.get("cik") or meta.get("EntityCentralIndexKey") or "UNKNOWN"
    form_type = meta.get("formType") or "10-K"
    accession = meta.get("accessionNo") or meta.get("accession") or stable_hash(folder.name)
    filed_at = meta.get("filedAt")

    # Try to load xbrl json
    xbrl_json = {}
    if files["xbrl"]:
        xj = load_json(files["xbrl"])
        # If it is nested under xbrl_data, keep it
        xbrl_json = xj

    # Load HTML/TXT narrative
    html_text = ""
    if files["html"]:
        html_text = files["html"].read_text(encoding="utf-8", errors="ignore")
    elif files["txt"]:
        html_text = files["txt"].read_text(encoding="utf-8", errors="ignore")

    # Build fact records
    fact_records: List[FactRecord] = []
    if xbrl_json:
        root = xbrl_json.get("xbrl_data", xbrl_json)
        fact_records = build_fact_records(root, cik=cik, accession=accession, source_path=str(files["xbrl"] or ""))

    # Parse narrative into sections/sentences
    sections: List[Tuple[str, str, List[str]]] = []
    if html_text:
        sections = extract_sections_and_sentences_from_html(html_text)

    # Connect to Neo4j
    uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    apply_schema(driver)

    with driver.session() as session:
        # Organization & Filing
        session.run(
            """
            MERGE (o:Organization {cik: $cik})
            ON CREATE SET o.name = $name
            ON MATCH SET o.name = coalesce(o.name, $name)
            MERGE (f:Filing {accession: $accession})
            SET f.formType = $formType, f.filedAt = $filedAt, f.sourceFolder = $folder, f.sourceMeta = $sourceMeta
            MERGE (o)-[:FILED]->(f)
            """,
            cik=cik, name=company_name, accession=accession, formType=form_type, filedAt=filed_at,
            folder=str(folder), sourceMeta=json.dumps(meta)[:100000]
        )

        # Securities from CoverPage fields if present
        securities = []
        cover = xbrl_json.get("xbrl_data", {}).get("CoverPage", {}) if xbrl_json else {}
        titles = cover.get("Security12bTitle", []) if isinstance(cover, dict) else []
        symbols = cover.get("TradingSymbol", []) if isinstance(cover, dict) else []
        exchanges = cover.get("SecurityExchangeName", []) if isinstance(cover, dict) else []

        def normalize_list(x):
            return x if isinstance(x, list) else []

        titles = normalize_list(titles)
        symbols = normalize_list(symbols)
        exchanges = normalize_list(exchanges)

        # Build tuples by index (best-effort)
        for i in range(max(len(titles), len(symbols), len(exchanges))):
            t = titles[i]["value"] if i < len(titles) and isinstance(titles[i], dict) else None
            sym = symbols[i]["value"] if i < len(symbols) and isinstance(symbols[i], dict) else None
            ex = exchanges[i]["value"] if i < len(exchanges) and isinstance(exchanges[i], dict) else None
            if t or sym or ex:
                securities.append({"title": t, "symbol": sym, "exchange": ex})

        if securities:
            session.run(
                """
                UNWIND $rows AS r
                MATCH (o:Organization {cik: $cik})
                MERGE (s:Security {title: coalesce(r.title, r.symbol, "Security")})
                SET s.symbol = r.symbol, s.exchange = r.exchange
                MERGE (o)-[:LISTED_SECURITY]->(s)
                WITH s, r
                FOREACH (ex IN CASE WHEN r.exchange IS NULL THEN [] ELSE [r.exchange] END |
                    MERGE (e:Exchange {code: ex})
                    MERGE (s)-[:ON_EXCHANGE]->(e)
                )
                """,
                cik=cik, rows=securities
            )

        # Facts (batch)
        if fact_records:
            rows = [{
                "id": fr.id,
                "concept": fr.concept,
                "value": fr.value,
                "value_type": fr.value_type,
                "unit": fr.unit,
                "decimals": fr.decimals,
                "period": {"id": fr.period.id, "start": fr.period.start, "end": fr.period.end, "instant": fr.period.instant},
                "context_ref": fr.context_ref,
                "dimensions": [{"axis": d.axis, "member": d.member} for d in fr.dimensions],
                "source_path": fr.source_path,
                "accession": accession,
            } for fr in fact_records]

            session.run(
                """
                UNWIND $rows AS r
                MATCH (f:Filing {accession: r.accession})
                MERGE (fact:Fact {id: r.id})
                ON CREATE SET fact.value = r.value, fact.valueType = r.value_type, fact.sourcePath = r.source_path, fact.decimals = r.decimals
                ON MATCH SET fact.value = r.value, fact.valueType = r.value_type, fact.sourcePath = r.source_path, fact.decimals = r.decimals
                MERGE (c:Concept {name: r.concept})
                MERGE (fact)-[:OF_CONCEPT]->(c)
                MERGE (f)-[:HAS_FACT]->(fact)
                WITH fact, r
                MERGE (p:Period {id: r.period.id})
                ON CREATE SET p.start = r.period.start, p.end = r.period.end, p.instant = r.period.instant
                ON MATCH SET p.start = r.period.start, p.end = r.period.end, p.instant = r.period.instant
                MERGE (fact)-[:FOR_PERIOD]->(p)
                FOREACH (u IN CASE WHEN r.unit IS NULL THEN [] ELSE [r.unit] END |
                    MERGE (unit:Unit {measure: u})
                    MERGE (fact)-[:MEASURED_IN]->(unit)
                )
                FOREACH (cr IN CASE WHEN r.context_ref IS NULL THEN [] ELSE [r.context_ref] END |
                    MERGE (ctx:Context {id: cr})
                    MERGE (fact)-[:IN_CONTEXT]->(ctx)
                )
                FOREACH (d IN r.dimensions |
                    MERGE (dim:Dimension {axis: d.axis, member: d.member})
                    MERGE (fact)-[:HAS_DIMENSION]->(dim)
                )
                """,
                rows=rows
            )

        # Narrative: Sections and Sentences
        if sections:
            # Bulk create sections
            section_rows = [{
                "item": item, "title": title, "accession": accession
            } for (item, title, _) in sections]
            session.run(
                """
                UNWIND $rows AS r
                MATCH (f:Filing {accession: r.accession})
                MERGE (sec:Section {filingAccession: r.accession, item: r.item, title: r.title})
                MERGE (f)-[:HAS_SECTION]->(sec)
                """,
                rows=section_rows
            )
            # Sentences
            sent_rows = []
            for (item, title, sents) in sections:
                for order, s in enumerate(sents):
                    sid = stable_hash(accession, item, str(order), s)  # deterministic
                    sent_rows.append({"id": sid, "text": s, "item": item, "title": title, "accession": accession, "order": order})
            if sent_rows:
                session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (sec:Section {filingAccession: r.accession, item: r.item, title: r.title})
                    MERGE (s:Sentence {id: r.id})
                    ON CREATE SET s.text = r.text, s.order = r.order
                    ON MATCH SET s.text = r.text, s.order = r.order
                    MERGE (sec)-[:HAS_SENTENCE]->(s)
                    """,
                    rows=sent_rows
                )

            # Optional topics (very naive)
            if extract_topics_flag:
                all_sents = [s["text"] for s in sent_rows]
                topics = extract_topics(all_sents, top_k=50)
                if topics:
                    session.run(
                        """
                        UNWIND $topics AS name
                        MERGE (t:Topic {name: name})
                        WITH collect(t) as ts
                        MATCH (f:Filing {accession: $accession})-[:HAS_SECTION]->(:Section)-[:HAS_SENTENCE]->(s:Sentence)
                        WITH ts, collect(s) as sents
                        // link each sentence to topics if sentence contains the topic substring (case-insensitive)
                        UNWIND sents AS sn
                        UNWIND ts AS tp
                        WITH sn, tp WHERE apoc.text.containsAll(toLower(sn.text), [toLower(tp.name)])  // requires APOC, otherwise fallback below
                        MERGE (sn)-[:MENTIONS]->(tp)
                        """,
                        topics=topics, accession=accession
                    )
                # If APOC not available, create a simpler linkage (best-effort)
                # Users can comment the APOC query and uncomment the below, but it will be slower:
                # for t in topics:
                #     session.run(
                #         """
                #         MATCH (s:Sentence) WHERE toLower(s.text) CONTAINS toLower($t)
                #         MERGE (tp:Topic {name: $t})
                #         MERGE (s)-[:MENTIONS]->(tp)
                #         """,
                #         t=t
                #     )

    driver.close()
    print(f"Ingestion complete for folder: {folder}")

def ingest_multiple_folders(root_folder: Path, extract_topics_flag: bool = False):
    """
    Find and ingest all 10-K folders within the given root folder.
    """
    print(f"Searching for 10-K folders in: {root_folder}")
    folders = find_10k_folders(root_folder)
    
    if not folders:
        print(f"No 10-K folders found in {root_folder}")
        return
    
    print(f"Found {len(folders)} 10-K folders to process:")
    for folder in folders:
        print(f"  - {folder}")
    
    failed_folders = []
    successful_folders = []
    
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] Processing: {folder}")
        try:
            ingest_folder(folder, extract_topics_flag)
            successful_folders.append(folder)
            print(f"✓ Successfully processed: {folder}")
        except Exception as e:
            print(f"✗ Failed to process {folder}: {e}")
            failed_folders.append((folder, str(e)))
    
    print(f"\n=== Processing Summary ===")
    print(f"Successfully processed: {len(successful_folders)} folders")
    print(f"Failed to process: {len(failed_folders)} folders")
    
    if failed_folders:
        print(f"\nFailed folders:")
        for folder, error in failed_folders:
            print(f"  - {folder}: {error}")
    
    if successful_folders:
        print(f"\nSuccessfully processed folders:")
        for folder in successful_folders:
            print(f"  - {folder}")

# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ingest 10-K folder(s) into Neo4j")
    ap.add_argument("folder", type=str, help="Folder containing 10-K files or parent folder with multiple 10-K subfolders")
    ap.add_argument("--extract-topics", action="store_true", help="Also extract naive Topic nodes from narrative")
    ap.add_argument("--recursive", "-r", action="store_true", 
                    help="Recursively process all 10-K folders found within the given folder")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder}")

    if args.recursive:
        ingest_multiple_folders(folder, extract_topics_flag=args.extract_topics)
    else:
        # Check if this folder contains 10-K files directly, or if it should be processed recursively
        files = guess_files(folder)
        if any(files.values()):
            # Single folder with 10-K files
            ingest_folder(folder, extract_topics_flag=args.extract_topics)
        else:
            # No 10-K files found directly, check if there are subfolders with 10-K files
            subfolders = find_10k_folders(folder)
            if subfolders:
                print(f"No 10-K files found directly in {folder}, but found {len(subfolders)} subfolder(s) with 10-K files.")
                print("Use --recursive flag to process all subfolders, or specify a specific subfolder.")
                print("Found subfolders:")
                for subfolder in subfolders:
                    print(f"  - {subfolder}")
                raise SystemExit(1)
            else:
                raise SystemExit(f"No 10-K files found in {folder} or its subfolders")

if __name__ == "__main__":
    main()
