#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_query.py

An MCP-capable querying tool (and CLI) that:
  1) Accepts a natural-language prompt.
  2) Retrieves the most relevant context from a local Neo4j graph.
  3) Calls a local LLM (defaults to gpt-oss-20b) with the prompt + retrieved context.
  4) Returns the model's answer and the context used.

Two modes:
  • CLI (default):   python mcp_kg_query.py --prompt "..." [--show-context]
  • MCP server:      python mcp_kg_query.py --mcp       (exposes a tool named "ask")

Environment variables (with sensible defaults):
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=neo4j_password
  NEO4J_DB=neo4j

Local model (defaults to Ollama):
  LLM_BACKEND=ollama | openai
  OLLAMA_BASE=http://localhost:11434
  OLLAMA_MODEL=gpt-oss:20b
  OPENAI_BASE= (optional) e.g., http://localhost:8000/v1  (for OpenAI-compatible servers)
  OPENAI_API_KEY= (if needed)
  OPENAI_MODEL= (e.g., llama-3.1-8b-instruct)

Requires:
  pip install neo4j requests
  (optional for MCP): pip install mcp

Notes:
  • Fulltext index "sentenceText" is used if present; otherwise falls back to substring scan.
  • Robust to any org (not IBM-specific). Tries to resolve a CIK from a ticker in the prompt.

How to run: 
    set the environment variables for the Neo4j database:
        export NEO4J_URI=bolt://localhost:7687
        export NEO4J_USER=neo4j
        export NEO4J_PASSWORD=your_password
        export NEO4J_DB=neo4j

    set the environment variables for the local model (Ollama) if not default:
        export LLM_BACKEND=ollama
        export OLLAMA_BASE=http://localhost:11434
        export OLLAMA_MODEL=gpt-oss:20b

python kg_query.py --prompt "I would like to run the following analysis on International Business Machines's (Ticker: IBM) business strategy. 1. Descriptive - Past performance - what happened and why? 2. Predictive - What will happen if the organization stays the current course? 3. Prescriptive - Things that should be considered to create a better future for the organization." --show-context
"""
import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests  # type: ignore
from neo4j import GraphDatabase  # type: ignore

# ---------------------------
# Utilities
# ---------------------------

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to was were will with without within into over under between during including until while than then so such via""".split())


def normspace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def stable_hash(*parts: str) -> str:
    import hashlib
    h = hashlib.md5("|".join([p if p is not None else "" for p in parts]).encode("utf-8")).hexdigest()
    return h[:40]

def pick_tokens(prompt: str, k: int = 12) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", prompt)
    toks = [w.lower() for w in words if w.lower() not in STOPWORDS and len(w) > 2]
    # keep order, dedupe
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out

# ---------------------------
# Neo4j Retrieval
# ---------------------------

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"

class Neo4jRetriever:
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

    def close(self):
        self.driver.close()

    # ---- Organization resolution (from ticker/org name) ----
    def resolve_org_from_prompt(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Try to resolve (cik, name) using uppercase ticker symbols or company substrings in the prompt.
        Returns (cik, name) or (None, None) if ambiguous.
        """
        ticker_candidates = re.findall(r"\b[A-Z]{1,5}\b", prompt)
        with self.driver.session(database=self.cfg.database) as session:
            # Try tickers first
            for sym in ticker_candidates:
                print(f"Trying ticker: {sym}")
                res = session.run(
                    """
                    MATCH (o:Organization)-[:LISTED_SECURITY]->(s:Security)
                    WHERE toUpper(s.symbol) = $sym
                    RETURN o.cik AS cik, o.name AS name LIMIT 1
                    """,
                    sym=sym,
                ).single()
                if res:
                    print(f"Resolved org: {res['name']} (CIK: {res['cik']})")
                    return res["cik"], res["name"]

            # Try org name fuzzy (first quoted phrase or longest capitalized phrase)
            m = re.search(r'"([^"]+)"', prompt)
            name_q = m.group(1) if m else None
            if not name_q:
                # naive: take longest sequence of capitalized words
                caps = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,4})\b", prompt)
                caps = sorted(caps, key=len, reverse=True)
                name_q = caps[0] if caps else None
            if name_q:
                res = session.run(
                    """
                    MATCH (o:Organization)
                    WHERE toLower(o.name) CONTAINS toLower($q)
                    RETURN o.cik AS cik, o.name AS name
                    ORDER BY size(o.name) ASC LIMIT 1
                    """,
                    q=name_q,
                ).single()
                if res:
                    return res["cik"], res["name"]
        return None, None

    # ---- Sentence fulltext (fallback aware) ----
    def top_sentences(self, query: str, cik: Optional[str], k: int = 40) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.cfg.database) as session:
            try:
                q = """
                CALL db.index.fulltext.queryNodes("sentenceText", $q) YIELD node, score
                OPTIONAL MATCH (sec:Section)-[:HAS_SENTENCE]->(node)
                OPTIONAL MATCH (f:Filing {accession: sec.filingAccession})
                OPTIONAL MATCH (o:Organization)-[:FILED]->(f)
                WITH node, score, sec, f, o
                WHERE $cik IS NULL OR o.cik = $cik
                RETURN node.id AS id, node.text AS text, score,
                       sec.item AS item, sec.title AS title,
                       f.accession AS accession, f.filedAt AS filedAt,
                       o.cik AS cik, o.name AS org
                ORDER BY score DESC
                LIMIT $k
                """
                rows = session.run(q, q=query, cik=cik, k=k).data()
                if rows:
                    return rows
            except Exception:
                pass
            # Fallback: substring scan with simple OR across tokens
            toks = pick_tokens(query, k=6)
            if not toks:
                toks = [query]
            where = " OR ".join([f"toLower(s.text) CONTAINS toLower($t{i})" for i in range(len(toks))])
            params = {f"t{i}": t for i, t in enumerate(toks)}
            params.update({"cik": cik, "k": k})
            q = f"""
            MATCH (s:Sentence)
            WHERE {where}
            OPTIONAL MATCH (sec:Section)-[:HAS_SENTENCE]->(s)
            OPTIONAL MATCH (f:Filing {{accession: sec.filingAccession}})
            OPTIONAL MATCH (o:Organization)-[:FILED]->(f)
            WITH s, sec, f, o
            WHERE $cik IS NULL OR o.cik = $cik
            RETURN s.id AS id, s.text AS text,
                   0.5 AS score,
                   sec.item AS item, sec.title AS title,
                   f.accession AS accession, f.filedAt AS filedAt,
                   o.cik AS cik, o.name AS org
            LIMIT $k
            """
            rows = session.run(q, **params).data()
            return rows

    # ---- Fact retrieval by concept-name match ----
    def top_facts(self, tokens: List[str], cik: Optional[str], limit: int = 60) -> List[Dict[str, Any]]:
        if not tokens:
            return []
        with self.driver.session(database=self.cfg.database) as session:
            q = """
            WITH $tokens AS toks
            UNWIND toks AS tok
            MATCH (c:Concept) WHERE toLower(c.name) CONTAINS toLower(tok)
            MATCH (f:Filing)-[:HAS_FACT]->(fact:Fact)-[:OF_CONCEPT]->(c)
            OPTIONAL MATCH (fact)-[:MEASURED_IN]->(u:Unit)
            OPTIONAL MATCH (fact)-[:FOR_PERIOD]->(p:Period)
            OPTIONAL MATCH (fact)-[:HAS_DIMENSION]->(d:Dimension)
            OPTIONAL MATCH (o:Organization)-[:FILED]->(f)
            WITH tok, c, fact, f, u, p, collect(DISTINCT {axis: d.axis, member: d.member}) AS dims, o
            WHERE $cik IS NULL OR o.cik = $cik
            RETURN tok, c.name AS concept, f.filedAt AS filedAt, f.accession AS accession,
                   fact.value AS value, u.measure AS unit,
                   p.start AS start, p.end AS end, p.instant AS instant,
                   dims
            ORDER BY filedAt DESC
            LIMIT $limit
            """
            rows = session.run(q, tokens=tokens, cik=cik, limit=limit).data()
            return rows

    def filing_summary(self, cik: Optional[str], limit: int = 10) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.cfg.database) as session:
            q = """
            MATCH (o:Organization)-[:FILED]->(f:Filing)
            WHERE $cik IS NULL OR o.cik = $cik
            RETURN o.cik AS cik, o.name AS org, f.accession AS accession, f.formType AS formType, f.filedAt AS filedAt
            ORDER BY f.filedAt DESC
            LIMIT $limit
            """
            return session.run(q, cik=cik, limit=limit).data()

# ---------------------------
# Context building
# ---------------------------

def build_context(prompt: str, cik: Optional[str], org_name: Optional[str],
                  sentences: List[Dict[str, Any]], facts: List[Dict[str, Any]], filings: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("<<PROMPT>>")
    lines.append(prompt.strip())
    lines.append("")

    if cik or org_name:
        lines.append("<<ORGANIZATION>>")
        lines.append(f"name: {org_name or 'Unknown'}")
        lines.append(f"cik: {cik or 'Unknown'}")
        lines.append("")

    if filings:
        lines.append("<<FILINGS (most recent)>>")
        for row in filings[:8]:
            lines.append(f"- {row.get('filedAt','?')} {row.get('formType','?')} accession {row.get('accession','?')}")
        lines.append("")

    if facts:
        lines.append("<<FACTS (recent, concept-matched)>>")
        # group by concept
        by_concept: Dict[str, List[Dict[str, Any]]] = {}
        for r in facts:
            by_concept.setdefault(r["concept"], []).append(r)
        # limit per concept
        for concept, rs in list(by_concept.items())[:12]:
            lines.append(f"- {concept}:")
            for r in rs[:3]:
                dims_str = ", ".join([f"{d.get('axis')}={d.get('member')}" for d in r.get("dims") or [] if d.get('axis') and d.get('member')])
                p = r.get("instant") or r.get("end") or r.get("start") or "?"
                unit = r.get("unit") or ""
                lines.append(f"    • {r.get('value')} {unit} @ {p} (filed {r.get('filedAt','?')} acc {r.get('accession','?')})" + (f" [{dims_str}]" if dims_str else ""))
        lines.append("")

    if sentences:
        lines.append("<<NARRATIVE SENTENCES (top)>>")
        for s in sentences[:20]:
            item = s.get("item") or ""
            filed = s.get("filedAt") or ""
            acc = s.get("accession") or ""
            lines.append(f"- [Item {item}] {filed} acc {acc}: {normspace(s.get('text',''))}")
        lines.append("")

    # final guide to model
    lines.append("<<INSTRUCTIONS>>")
    lines.append(
        "Using the context above, answer the user's prompt. "
        "Cite accession numbers or items when relevant. "
        "If data is insufficient, state assumptions. Be concise but specific."
    )
    return "\n".join(lines)

# ---------------------------
# Local LLM adapters
# ---------------------------

class LLMBase:
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError

class OllamaLLM(LLMBase):
    def __init__(self, base: str, model: str, timeout: int = 120):
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        url = f"{self.base}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # ollama responses may be in 'message' or 'messages'
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")
        if "messages" in data and isinstance(data["messages"], list) and data["messages"]:
            return data["messages"][-1].get("content", "")
        # fallback to /api/generate if /api/chat not supported
        url = f"{self.base}/api/generate"
        payload = {
            "model": self.model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "")

class OpenAICompatLLM(LLMBase):
    def __init__(self, base: str, model: str, api_key: Optional[str] = None, timeout: int = 120):
        self.base = base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        url = f"{self.base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

def make_llm_from_env() -> LLMBase:
    backend = os.getenv("LLM_BACKEND", "ollama").lower().strip()
    if backend == "openai":
        base = os.getenv("OPENAI_BASE", "http://localhost:8000/v1")
        model = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instruct")
        key = os.getenv("OPENAI_API_KEY")
        return OpenAICompatLLM(base, model, key)
    # default: ollama
    base = os.getenv("OLLAMA_BASE", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    return OllamaLLM(base, model)

# ---------------------------
# Query pipeline
# ---------------------------

def run_pipeline(prompt: str, cfg: Neo4jConfig, show_context: bool = False) -> Dict[str, Any]:
    retriever = Neo4jRetriever(cfg)
    try:
        cik, org_name = retriever.resolve_org_from_prompt(prompt)
        toks = pick_tokens(prompt, k=12)
        sentences = retriever.top_sentences(prompt, cik=cik, k=40)
        facts = retriever.top_facts(tokens=toks, cik=cik, limit=80)
        filings = retriever.filing_summary(cik=cik, limit=8)

        ctx = build_context(prompt, cik, org_name, sentences, facts, filings)

        llm = make_llm_from_env()
        system = "You are a strategic planning assistant using financial and disclosure-analysis to provide insights and recommendations for organizations. Your answer should be augmented and driven by the supplied context when possible. Respond is cleaning defined sections, do not use tables."
        answer = llm.generate(system, ctx)

        if show_context:
            return {"answer": answer, "context": ctx, "cik": cik, "org": org_name}
        return {"answer": answer, "cik": cik, "org": org_name}
    finally:
        retriever.close()

# ---------------------------
# CLI
# ---------------------------

def cli_main(args: argparse.Namespace) -> int:
    cfg = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        database=os.getenv("NEO4J_DB", "neo4j"),
    )
    if not args.prompt:
        print("Enter your prompt (end with Ctrl-D):")
        prompt = sys.stdin.read().strip()
    else:
        prompt = args.prompt.strip()

    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return 2

    result = run_pipeline(prompt, cfg, show_context=args.show_context)
    print("\n=== ANSWER ===\n")
    print(result["answer"].strip())
    if args.show_context and "context" in result:
        print("\n=== CONTEXT USED ===\n")
        print(result["context"])
    return 0

# ---------------------------
# MCP server (optional)
# ---------------------------

def mcp_main() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as e:
        print("The 'mcp' package is required for --mcp mode. Try: pip install mcp", file=sys.stderr)
        return 2

    app = FastMCP("strategery-kg")

    @app.tool()
    def ask(prompt: str, show_context: bool = False) -> dict:
        """
        Query the Strategery knowledge graph and a local model.

        Args:
            prompt: The user's question.
            show_context: If true, returns the retrieved context along with the answer.
        Returns: dict with 'answer' and optionally 'context'.
        """
        cfg = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
            database=os.getenv("NEO4J_DB", "neo4j"),
        )
        return run_pipeline(prompt, cfg, show_context=show_context)

    app.run()
    return 0

# ---------------------------
# Entrypoint
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MCP/CLI Querying Tool for Strategery Neo4j Graph")
    ap.add_argument("--prompt", type=str, help="Natural-language question to ask")
    ap.add_argument("--show-context", action="store_true", help="Print the retrieved context")
    ap.add_argument("--mcp", action="store_true", help="Run as an MCP tool server instead of CLI")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.mcp:
        sys.exit(mcp_main())
    else:
        sys.exit(cli_main(args))

if __name__ == "__main__":
    main()