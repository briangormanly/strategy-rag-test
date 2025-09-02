from sec_api import QueryApi, XbrlApi
import os
import re
import json
import hashlib
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# ---------- Config ----------
NUM_YEARS = 5  # how many most-recent 10-K filings to pull
SEC_HEADERS = {
    # Per SEC guidance, identify yourself. Replace with your contact info.
    "User-Agent": "Research script (your_email@example.com)"
}
REQUEST_TIMEOUT = 30

# ---------- Helpers ----------
def load_api_key():
    try:
        with open('api_key.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("api_key.txt file not found. Please create this file with your SEC API key.")

queryApi = QueryApi(api_key=load_api_key())
xbrlApi = XbrlApi(api_key=load_api_key())

search_params = {
    "query": "ticker:IBM AND formType:\"10-K\"",
    "from": "0",
    "size": str(NUM_YEARS),
    "sort": [{"filedAt": {"order": "desc"}}]
}

def get_cache_filename(search_params):
    params_str = json.dumps(search_params, sort_keys=True)
    hash_obj = hashlib.md5(params_str.encode())
    return f"query_{hash_obj.hexdigest()}.json"

def is_cache_valid(path, days=7):
    if not os.path.exists(path):
        return False
    file_time = datetime.fromtimestamp(os.path.getmtime(path))
    cutoff_time = datetime.now() - timedelta(days=days)
    return file_time > cutoff_time

def extract_year_from_filing_date(filed_at):
    try:
        filing_year = int(filed_at.split('-')[0])
        # 10-K filings are typically filed in the year after the fiscal year they report
        # So subtract 1 to get the fiscal year
        return str(filing_year - 1)
    except Exception:
        return "unknown_year"

def get_company_folder_name(company_name):
    safe = "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in (company_name or 'unknown_company'))
    return safe.replace(' ', '_').strip('_') or 'unknown_company'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def filing_paths(filing, year=None):
    year = year or extract_year_from_filing_date(filing.get("filedAt", ""))
    company = get_company_folder_name(filing.get("companyName", "unknown_company"))
    company_dir = os.path.join("10k", company)
    year_dir = os.path.join(company_dir, year)
    ensure_dir(year_dir)
    base = os.path.join(year_dir, f"{year}_10k")
    return {
        "xbrl_json": base + ".json",
        "primary_html": base + "_primary.html",
        "primary_txt": base + "_primary.txt",
        "meta": base + "_meta.json",
        "dir": year_dir,
        "base": base
    }

def extract_filing_fields(filing):
    return {
        "ticker": filing.get("ticker"),
        "formType": filing.get("formType"),
        "accessionNo": filing.get("accessionNo"),
        "cik": filing.get("cik"),
        "companyNameLong": filing.get("companyNameLong"),
        "companyName": filing.get("companyName"),
        "linkToFilingDetails": filing.get("linkToFilingDetails"),
        "description": filing.get("description"),
        "linkToTxt": filing.get("linkToTxt"),
        "filedAt": filing.get("filedAt")
    }

def save_search_cache(response, cache_file):
    simplified = [extract_filing_fields(f) for f in response.get("filings", [])]
    with open(cache_file, 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "total": response.get("total", {}),
                   "filings": simplified}, f, indent=2)

def load_search_cache(cache_file):
    with open(cache_file, 'r') as f:
        d = json.load(f)
    return {"total": d.get("total", {}), "filings": d.get("filings", [])}

# ---------- HTML → Text cleaning ----------
TEXTBLOCK_SUFFIXES = ("TextBlock", "PolicyTextBlock", "TableTextBlock", "NarrativeTextBlock")

def strip_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove non-content tags
    for t in soup(["style", "script", "noscript"]):
        t.decompose()
    # Unwrap common presentational containers; drop attributes to reduce noise
    for tag in soup.find_all(True):
        if tag.name in {"span", "font", "div"}:
            tag.unwrap()
        else:
            tag.attrs = {}
    text = soup.get_text("\n", strip=True)
    # normalize whitespace
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def clean_textblocks(obj):
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if isinstance(v, str) and any(k.endswith(suf) for suf in TEXTBLOCK_SUFFIXES) and "<" in v and ">" in v:
                cleaned[k.replace("TextBlock", "Text")] = strip_html_to_text(v)
            else:
                cleaned[k] = clean_textblocks(v)
        return cleaned
    if isinstance(obj, list):
        return [clean_textblocks(x) for x in obj]
    return obj

# ---------- Primary iXBRL discovery ----------
def derive_folder_index_json_url(filing):
    """
    Try to build .../<accession-no>/index.json from either linkToFilingDetails or linkToTxt.
    """
    details = filing.get("linkToFilingDetails") or ""
    txt = filing.get("linkToTxt") or ""
    # Common patterns:
    #   ...-index.htm  -> ...-index.json
    #   ...-index.html -> ...-index.json
    if details:
        u = (details
             .replace("-index.htm", "-index.json")
             .replace("-index.html", "-index.json"))
        if u.endswith(".json"):
            return u
    # Fallback: /.../<accession>.txt  ->  /.../<accession>/index.json
    if txt and txt.endswith(".txt"):
        return txt[:-4] + "/index.json"
    return None

def fetch_json(url):
    r = requests.get(url, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def fetch_text(url):
    r = requests.get(url, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

def pick_primary_html_from_index(index_json, index_json_url):
    """
    SEC folder index.json lists all files. Choose the primary 10-K HTML (prefer iXBRL).
    """
    items = (index_json.get("directory", {}) or {}).get("item", []) or []
    base = index_json_url.rsplit("/", 1)[0]  # folder URL
    # Candidates: TYPE == '10-K' and .htm(l)
    def is_html(name):
        return name.lower().endswith((".htm", ".html"))
    # Prefer items explicitly typed as 10-K
    typed = [it for it in items if it.get("type", "").upper() in {"10-K", "10K", "10-K405"} and is_html(it.get("name",""))]
    candidates = typed if typed else [it for it in items if is_html(it.get("name",""))]
    if not candidates:
        return None, None
    # Heuristic: choose the largest by size (often the full HTML)
    best = max(candidates, key=lambda it: int(it.get("size", "0") or 0))
    url = f"{base}/{best.get('name')}"
    html = fetch_text(url)
    # If not iXBRL, see if any other candidate contains '<ix:' (rare but safe)
    if "<ix:" not in html and len(candidates) > 1:
        for it in sorted(candidates, key=lambda x: int(x.get("size","0") or 0), reverse=True):
            alt_url = f"{base}/{it.get('name')}"
            alt_html = fetch_text(alt_url)
            if "<ix:" in alt_html:
                return alt_url, alt_html
    return url, html

def iter_submission_documents(submission_txt):
    """
    Yields dicts with keys: type, filename, text (raw inside <TEXT>...</TEXT>).
    """
    for m in re.finditer(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>", submission_txt):
        block = m.group(1)
        doc_type = re.search(r"(?im)^\s*<TYPE>\s*([^\r\n<]+)", block)
        filename = re.search(r"(?im)^\s*<FILENAME>\s*([^\r\n<]+)", block)
        text_match = re.search(r"(?is)<TEXT>(.*?)</TEXT>", block)
        yield {
            "type": (doc_type.group(1).strip() if doc_type else "").upper(),
            "filename": (filename.group(1).strip() if filename else ""),
            "text": text_match.group(1) if text_match else ""
        }

def pick_primary_html_from_submission_txt(link_to_txt):
    """
    Parse the .txt submission to locate the 10-K HTML block; prefer iXBRL.
    """
    txt = fetch_text(link_to_txt)
    docs = list(iter_submission_documents(txt))
    # Primary candidates
    html_docs = [d for d in docs if d["filename"].lower().endswith((".htm", ".html"))]
    tenk_html = [d for d in html_docs if d["type"] in {"10-K", "10K", "10-K405"}] or html_docs
    if not tenk_html:
        return None, None
    # Prefer iXBRL by scanning for '<ix:' token; else choose largest length
    best = None
    for d in tenk_html:
        if "<ix:" in d["text"]:
            best = d
            break
    if best is None:
        best = max(tenk_html, key=lambda d: len(d["text"]))
    # Reconstruct a synthetic 'url' for meta; raw HTML content is in d["text"]
    return f"{link_to_txt}#/{best['filename']}", best["text"]

# ---------- Main ----------
try:
    ensure_dir("response")
    cache_path = os.path.join("response", get_cache_filename(search_params))

    if is_cache_valid(cache_path):
        print("Using cached search response (≤7 days old)")
        response = load_search_cache(cache_path)
    else:
        print("Fetching filings list…")
        response = queryApi.get_filings(search_params)
        save_search_cache(response, cache_path)
        response = load_search_cache(cache_path)
        print(f"Cached search to: {cache_path}")

    print(f"Total filings matching: {response['total'].get('value')}")
    print("\nRetrieved Filings:")
    for f in response["filings"]:
        print(f"  Accession No: {f['accessionNo']}")
        print(f"  Form Type: {f['formType']}")
        print(f"  Filed At: {f['filedAt']}")
        print(f"  Company: {f['companyName']} (CIK {f['cik']})")
        print(f"  Details: {f['linkToFilingDetails']}")
        print(f"  TXT:     {f['linkToTxt']}")
        print("--------------------------------")

    print("\nProcessing filings…")
    for i, filing in enumerate(response["filings"], 1):
        year_guess = extract_year_from_filing_date(filing.get("filedAt",""))
        paths = filing_paths(filing, year_guess)

        print(f"\n[{i}/{len(response['filings'])}] {filing['companyName']} {filing['accessionNo']} ({year_guess})")
        primary_html = None
        primary_url = None

        # 1) Try folder index.json to locate primary iXBRL HTML
        try:
            idx_url = derive_folder_index_json_url(filing)
            if idx_url:
                idx_json = fetch_json(idx_url)
                primary_url, primary_html = pick_primary_html_from_index(idx_json, idx_url)
        except Exception as e:
            print(f"  index.json method failed: {e}")

        # 2) Fallback: parse linkToTxt submission
        if not primary_html:
            try:
                lt = filing.get("linkToTxt")
                if lt:
                    primary_url, primary_html = pick_primary_html_from_submission_txt(lt)
            except Exception as e:
                print(f"  linkToTxt fallback failed: {e}")

        if not primary_html:
            print("  Could not locate primary 10-K HTML; skipping text extraction for this filing.")
        else:
            # Save raw primary HTML (cache-aware)
            if not is_cache_valid(paths["primary_html"]) or os.path.getsize(paths["primary_html"]) == 0:
                with open(paths["primary_html"], "w", encoding="utf-8") as f:
                    f.write(primary_html)
                print(f"  Saved primary HTML → {paths['primary_html']}")
            else:
                print(f"  Using cached primary HTML → {paths['primary_html']}")

            # Save cleaned plain-text for search/IR
            cleaned = strip_html_to_text(primary_html)
            with open(paths["primary_txt"], "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"  Saved cleaned text → {paths['primary_txt']} (chars: {len(cleaned):,})")

        # 3) XBRL: convert to JSON (use the primary HTML URL if we found one)
        try:
            if is_cache_valid(paths["xbrl_json"]):
                with open(paths["xbrl_json"], "r") as f:
                    xbrl_json = json.load(f)
                print(f"  Using cached XBRL JSON → {paths['xbrl_json']}")
            else:
                # Prefer calling with the primary HTML URL (better than details page)
                xbrl_src_url = primary_url or filing['linkToFilingDetails']
                print(f"  Converting XBRL from: {xbrl_src_url}")
                xbrl_json = xbrlApi.xbrl_to_json(htm_url=xbrl_src_url)
                xbrl_json = clean_textblocks(xbrl_json)
                with open(paths["xbrl_json"], "w") as f:
                    json.dump({"timestamp": datetime.now().isoformat(),
                               "xbrl_data": xbrl_json}, f, indent=2)
                print(f"  Saved XBRL JSON → {paths['xbrl_json']}")

            # If the CoverPage has DocumentFiscalYearFocus, consider renaming base files (optional)
            try:
                fy = (xbrl_json.get("xbrl_data", xbrl_json).get("CoverPage", {}) or {}).get("DocumentFiscalYearFocus")
                if fy and fy != year_guess:
                    # rename files to fiscal year
                    new_paths = filing_paths(filing, fy)
                    for key in ("xbrl_json", "primary_html", "primary_txt", "meta"):
                        if os.path.exists(paths[key]) and paths[key] != new_paths[key]:
                            os.replace(paths[key], new_paths[key])
                    paths = new_paths
                    print(f"  Renamed artifacts to fiscal year {fy}")
            except Exception:
                pass

        except Exception as e:
            print(f"  XBRL conversion failed: {e}")

        # 4) Write a small metadata file tying it all together
        meta = {
            "ticker": filing.get("ticker"),
            "company": filing.get("companyName"),
            "cik": filing.get("cik"),
            "formType": filing.get("formType"),
            "accessionNo": filing.get("accessionNo"),
            "filedAt": filing.get("filedAt"),
            "primary_html_url": primary_url,
            "artifacts": {
                "xbrl_json": paths["xbrl_json"],
                "primary_html": paths["primary_html"],
                "primary_txt": paths["primary_txt"]
            }
        }
        with open(paths["meta"], "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Wrote meta → {paths['meta']}")

except Exception as e:
    print(f"An error occurred: {e}")