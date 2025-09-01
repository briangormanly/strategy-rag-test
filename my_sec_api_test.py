from sec_api import QueryApi
import os
import json
import hashlib
from datetime import datetime, timedelta

# Read API key from file
def load_api_key():
    try:
        with open('api_key.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("api_key.txt file not found. Please create this file with your SEC API key.")

queryApi = QueryApi(api_key=load_api_key())

# Define search parameters
search_params = {
    "query": "ticker:IBM AND formType:\"10-K\"",  # Search for IBM's 10-K filings
    "from": "0",                                 # Start from the first result
    "size": "5",                                # Retrieve 2 filings per request
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

    

except Exception as e:
    print(f"An error occurred: {e}")
