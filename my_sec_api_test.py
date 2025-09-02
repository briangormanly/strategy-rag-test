from sec_api import QueryApi, XbrlApi
import os
import json
import hashlib
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Read API key from file
def load_api_key():
    try:
        with open('api_key.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("api_key.txt file not found. Please create this file with your SEC API key.")

queryApi = QueryApi(api_key=load_api_key())
xbrlApi = XbrlApi(api_key=load_api_key())

# Configuration: Number of years of 10-K filings to retrieve
numYears = 1  # This will get the last 5 years (e.g., 2024, 2023, 2022, 2021, 2020)

# Define search parameters
search_params = {
    "query": "ticker:IBM AND formType:\"10-K\"",  # Search for IBM's 10-K filings
    "from": "0",                                 # Start from the first result
    "size": str(numYears),                      # Retrieve numYears filings per request
    "sort": [{"filedAt": {"order": "desc"}}]    # Sort by filing date in descending order
}

def get_cache_filename(search_params):
    params_str = json.dumps(search_params, sort_keys=True)
    hash_obj = hashlib.md5(params_str.encode())
    return f"query_{hash_obj.hexdigest()}.json"

def is_cache_valid(cache_file, days=7):
    if not os.path.exists(cache_file):
        return False
    
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
    cutoff_time = datetime.now() - timedelta(days=days)
    
    return file_time > cutoff_time

def extract_filing_fields(filing):
    """Extract only the required fields from a filing object"""
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

def save_to_cache(response, cache_file):
    # Extract only the required fields from each filing
    simplified_filings = []
    for filing in response.get("filings", []):
        simplified_filings.append(extract_filing_fields(filing))
    
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "total": response.get("total", {}),
        "filings": simplified_filings
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

def load_from_cache(cache_file):
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    # Return the simplified data structure (no longer nested under "response")
    return {
        "total": cache_data.get("total", {}),
        "filings": cache_data.get("filings", [])
    }

def extract_year_from_filing_date(filed_at):
    """Extract year from filing date string like '2025-02-25T16:12:45-05:00'"""
    try:
        return filed_at.split('-')[0]
    except:
        return "unknown_year"

def get_company_folder_name(company_name):
    """Create a safe folder name from company name"""
    # Remove special characters and replace spaces with underscores
    safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in company_name)
    safe_name = safe_name.replace(' ', '_').strip('_')
    return safe_name

def get_xbrl_cache_path(filing):
    """Generate cache file path for XBRL data"""
    company_name = filing.get('companyName', 'unknown_company')

    fy = (xbrl_json.get("CoverPage", {}) or {}).get("DocumentFiscalYearFocus")
    year_for_filename = fy or extract_year_from_filing_date(filing.get("filedAt",""))

    company_folder = get_company_folder_name(company_name)
    cache_dir = os.path.join("10k", company_folder)
    
    # Create directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    filename = f"{year_for_filename}_10k.json"
    return os.path.join(cache_dir, filename)

def save_xbrl_to_cache(xbrl_data, cache_file_path):
    """Save XBRL JSON data to cache file"""
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "xbrl_data": xbrl_data
    }
    
    with open(cache_file_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

def load_xbrl_from_cache(cache_file_path):
    """Load XBRL JSON data from cache file"""
    with open(cache_file_path, 'r') as f:
        cache_data = json.load(f)
    return cache_data.get("xbrl_data", {})


TEXTBLOCK_SUFFIXES = ("TextBlock", "PolicyTextBlock", "TableTextBlock", "NarrativeTextBlock")

def strip_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # remove non-content tags
    for t in soup(["style", "script"]):
        t.decompose()
    # unwrap purely presentational containers and drop all attributes
    for tag in soup.find_all(True):
        if tag.name in {"span", "font", "div"}:
            tag.unwrap()
        else:
            tag.attrs = {}
    # collapse to readable text
    return soup.get_text("\n", strip=True)

def clean_textblocks(obj):
    """Recursively replace *TextBlock HTML with plain text; leave numeric facts alone."""
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

try:
    # Create response folder if it doesn't exist
    response_folder = "response"
    if not os.path.exists(response_folder):
        os.makedirs(response_folder)
    
    # Generate cache filename based on search parameters
    cache_filename = get_cache_filename(search_params)
    cache_file_path = os.path.join(response_folder, cache_filename)
    
    # Check if we have a valid cache (within last week)
    if is_cache_valid(cache_file_path):
        print("Using cached response (within last week)")
        response = load_from_cache(cache_file_path)
    else:
        print("Making API request...")
        # Execute the search query
        response = queryApi.get_filings(search_params)
        
        # Save response to cache
        save_to_cache(response, cache_file_path)
        print(f"Response cached to: {cache_file_path}")

    # Print the total number of matching filings
    print(f"Total filings matching the criteria: {response['total']['value']}")

    # Iterate and print details of each retrieved filing
    print("\nRetrieved Filings:")
    for filing in response["filings"]:
        print(f"  Accession No: {filing['accessionNo']}")
        print(f"  Form Type: {filing['formType']}")
        print(f"  Filed At: {filing['filedAt']}")
        print(f"  Company Name: {filing['companyName']}")
        print(f"  CIK: {filing['cik']}")
        print(f"  Company Name Long: {filing['companyNameLong']}")
        print(f"  Description: {filing['description']}")
        print(f"  Link to Filing Details: {filing['linkToFilingDetails']}")
        print(f"  Link to TXT: {filing['linkToTxt']}")
        print("--------------------------------")

    # Convert each 10-K filing to JSON using XbrlApi
    print("\nConverting 10-K filings to JSON...")
    for i, filing in enumerate(response["filings"], 1):
        try:
            # Generate cache path for this filing
            xbrl_cache_path = get_xbrl_cache_path(filing)
            year = extract_year_from_filing_date(filing.get('filedAt', ''))
            company_folder = get_company_folder_name(filing.get('companyName', 'unknown_company'))
            
            print(f"Processing filing {i}/{numYears}: {filing['accessionNo']} ({year})")
            print(f"Company: {filing['companyName']}")
            print(f"Cache path: {xbrl_cache_path}")
            
            # Check if we have valid cached XBRL data
            if is_cache_valid(xbrl_cache_path):
                print(f"Using cached XBRL data for {year} (within last week)")
                xbrl_json = load_xbrl_from_cache(xbrl_cache_path)
            else:
                print(f"Making XBRL API request for {year}...")
                url_10k = filing['linkToFilingDetails']
                print(f"URL: {url_10k}")
                
                # Convert XBRL to JSON
                xbrl_json = xbrlApi.xbrl_to_json(htm_url=url_10k)

                # Clean the XBRL JSON
                xbrl_json = clean_textblocks(xbrl_json)
                
                # Save to cache
                save_xbrl_to_cache(xbrl_json, xbrl_cache_path)
                print(f"XBRL data cached to: {xbrl_cache_path}")
            
            print(f"Successfully processed filing {filing['accessionNo']} ({year})")
            print(f"JSON keys available: {list(xbrl_json.keys()) if isinstance(xbrl_json, dict) else 'Not a dictionary'}")
            print("=" * 50)

            # print("--------------------------------")
            # print(xbrl_json)
            # print("--------------------------------")
            
        except Exception as filing_error:
            print(f"Error processing filing {filing['accessionNo']}: {filing_error}")
            print("=" * 50)
            continue

except Exception as e:
    print(f"An error occurred: {e}")
