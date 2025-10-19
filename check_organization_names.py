#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_organization_names.py

Check what organization names exist in the database.
"""

import os
from neo4j import GraphDatabase


def check_organization_names():
    """Check what organization names exist."""
    
    # Configuration
    cfg = {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        "database": os.getenv("NEO4J_DB", "neo4j"),
    }
    
    print("Checking Organization Names")
    print("=" * 30)
    
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    
    try:
        with driver.session(database=cfg["database"]) as session:
            
            # Get all organizations
            result = session.run("MATCH (o:Organization) RETURN o.name, o.cik ORDER BY o.name")
            orgs = list(result)
            
            print(f"Found {len(orgs)} organizations:")
            for record in orgs:
                name = record['o.name']
                cik = record['o.cik']
                print(f"  - '{name}' (CIK: {cik})")
            
            print()
            
            # Check for IBM variations
            print("Checking for IBM variations:")
            variations = ["IBM", "International Business Machines", "INTERNATIONAL BUSINESS MACHINES"]
            
            for variation in variations:
                result = session.run("MATCH (o:Organization) WHERE o.name CONTAINS $name RETURN o.name, o.cik", name=variation)
                matches = list(result)
                if matches:
                    print(f"  ✓ Found with '{variation}': {matches[0]['o.name']} (CIK: {matches[0]['o.cik']})")
                else:
                    print(f"  ✗ No match for '{variation}'")
            
            print()
            
            # Check CIK 51143 specifically
            print("Checking CIK 51143 (IBM's CIK):")
            result = session.run("MATCH (o:Organization {cik: '51143'}) RETURN o.name, o.cik")
            ibm_org = result.single()
            if ibm_org:
                print(f"  ✓ Found IBM: '{ibm_org['o.name']}' (CIK: {ibm_org['o.cik']})")
            else:
                print("  ✗ No organization with CIK 51143")
            
            # Check CIK 0000051143 (with leading zeros)
            print("Checking CIK 0000051143:")
            result = session.run("MATCH (o:Organization {cik: '0000051143'}) RETURN o.name, o.cik")
            ibm_org = result.single()
            if ibm_org:
                print(f"  ✓ Found IBM: '{ibm_org['o.name']}' (CIK: {ibm_org['o.cik']})")
            else:
                print("  ✗ No organization with CIK 0000051143")
            
            print()
            print("Based on the results above, we'll use the correct CIK to fix the links.")
            
    except Exception as e:
        print(f"Error checking organizations: {e}")
        return False
    
    finally:
        driver.close()
    
    return True


if __name__ == "__main__":
    check_organization_names()
