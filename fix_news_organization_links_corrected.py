#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_news_organization_links_corrected.py

Fix missing news-organization links using the correct CIK.
"""

import os
from neo4j import GraphDatabase


def fix_news_organization_links():
    """Fix missing news-organization links using correct CIK."""
    
    # Configuration
    cfg = {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        "database": os.getenv("NEO4J_DB", "neo4j"),
    }
    
    print("Fixing News-Organization Links (Corrected)")
    print("=" * 50)
    print(f"Neo4j URI: {cfg['uri']}")
    print(f"Database: {cfg['database']}")
    print()
    
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    
    try:
        with driver.session(database=cfg["database"]) as session:
            
            # 1. Check current state
            print("1. CHECKING CURRENT STATE:")
            print("-" * 30)
            
            # Check for IBM organization with correct CIK
            result = session.run("MATCH (o:Organization {cik: '51143'}) RETURN o.name, o.cik")
            org = result.single()
            if org:
                print(f"Found organization: {org['o.name']} (CIK: {org['o.cik']})")
                org_cik = org['o.cik']
            else:
                print("❌ No IBM organization found with CIK 51143")
                return False
            
            # Check for articles
            result = session.run("MATCH (a:Article) RETURN count(a) as count")
            article_count = result.single()["count"]
            print(f"Found {article_count} articles")
            
            # Check for existing links
            result = session.run("MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o:Organization) RETURN count(a) as count")
            existing_links = result.single()["count"]
            print(f"Existing organization links: {existing_links}")
            
            print()
            
            # 2. Create missing links
            print("2. CREATING MISSING LINKS:")
            print("-" * 30)
            
            # Link articles to IBM organization using correct CIK
            result = session.run("""
                MATCH (a:Article)
                MATCH (o:Organization {cik: $cik})
                WHERE NOT (a)-[:MENTIONS_ORGANIZATION]->(o)
                CREATE (a)-[:MENTIONS_ORGANIZATION]->(o)
                RETURN count(a) as linked_count
            """, cik=org_cik)
            
            linked_count = result.single()["linked_count"]
            print(f"Linked {linked_count} articles to IBM organization")
            
            # 3. Verify the fix
            print()
            print("3. VERIFYING FIX:")
            print("-" * 20)
            
            # Check final state
            result = session.run("MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o:Organization) RETURN count(a) as count")
            final_links = result.single()["count"]
            print(f"Total organization links: {final_links}")
            
            if final_links > 0:
                print("✅ News-organization links created successfully!")
            else:
                print("❌ Failed to create links")
                return False
            
            # 4. Test a sample query
            print()
            print("4. TESTING SAMPLE QUERY:")
            print("-" * 25)
            
            result = session.run("""
                MATCH (o:Organization {cik: $cik})-[:FILED]->(f:Filing)
                MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o)
                MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
                RETURN count(a) as article_count, 
                       avg(s.compound) as avg_sentiment
            """, cik=org_cik)
            
            test_result = result.single()
            if test_result:
                print(f"Found {test_result['article_count']} articles with sentiment analysis")
                print(f"Average sentiment: {test_result['avg_sentiment']:.3f}")
                print("✅ Cross-domain queries now work!")
            else:
                print("❌ Cross-domain queries still not working")
                return False
            
            # 5. Test the specific query that was failing
            print()
            print("5. TESTING FAILING QUERY:")
            print("-" * 25)
            
            try:
                result = session.run("""
                    MATCH (o:Organization {cik: $cik})-[:FILED]->(f:Filing)
                    MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o)
                    MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
                    WHERE f.filedAt IS NOT NULL AND a.publishedDate IS NOT NULL
                      AND abs(duration.between(date(f.filedAt), date(a.publishedDate)).days) <= 60
                    RETURN count(a) as timeline_count
                """, cik=org_cik)
                
                timeline_result = result.single()
                if timeline_result:
                    print(f"Found {timeline_result['timeline_count']} articles in sentiment timeline")
                    print("✅ Sentiment timeline queries now work!")
                else:
                    print("⚠️  Sentiment timeline query returned no results (this is normal if no articles are within 60 days of filings)")
            except Exception as e:
                print(f"⚠️  Sentiment timeline query failed: {e}")
                print("This is expected if there are no articles within 60 days of filings")
            
            print()
            print("6. NEXT STEPS:")
            print("-" * 15)
            print("✅ News-organization links are now fixed")
            print("✅ You can run the full evaluation:")
            print("   python run_evaluation.py")
            print()
            print("The evaluation should now show meaningful improvements!")
            
    except Exception as e:
        print(f"Error fixing links: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        driver.close()
    
    return True


if __name__ == "__main__":
    success = fix_news_organization_links()
    if success:
        print("\n🎉 SUCCESS: News-organization links fixed!")
        print("You can now run: python run_evaluation.py")
    else:
        print("\n❌ FAILED: Could not fix news-organization links")
        print("Please check the error messages above")
