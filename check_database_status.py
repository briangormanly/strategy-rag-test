#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_database_status.py

Diagnostic script to check what data exists in the Neo4j database
and identify what's missing for the evaluation framework.
"""

import os
import sys
from neo4j import GraphDatabase


def check_database_status():
    """Check what data exists in the Neo4j database."""
    
    # Configuration
    cfg = {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        "database": os.getenv("NEO4J_DB", "neo4j"),
    }
    
    print("Database Status Check")
    print("=" * 50)
    print(f"Neo4j URI: {cfg['uri']}")
    print(f"Database: {cfg['database']}")
    print()
    
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    
    try:
        with driver.session(database=cfg["database"]) as session:
            
            # 1. Check node types and counts
            print("1. NODE TYPES:")
            print("-" * 20)
            result = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
                RETURN label, value.count as count
                ORDER BY count DESC
            """)
            
            node_types = {}
            for record in result:
                label = record["label"]
                count = record["count"]
                node_types[label] = count
                print(f"  {label}: {count:,}")
            
            print()
            
            # 2. Check relationship types and counts
            print("2. RELATIONSHIP TYPES:")
            print("-" * 25)
            result = session.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {}) YIELD value
                RETURN relationshipType, value.count as count
                ORDER BY count DESC
            """)
            
            relationship_types = {}
            for record in result:
                rel_type = record["relationshipType"]
                count = record["count"]
                relationship_types[rel_type] = count
                print(f"  {rel_type}: {count:,}")
            
            print()
            
            # 3. Check for specific data needed for evaluation
            print("3. EVALUATION REQUIREMENTS CHECK:")
            print("-" * 35)
            
            # Check for 10-K data
            has_10k_data = "Organization" in node_types and "Filing" in node_types
            print(f"  ✓ 10-K Financial Data: {'YES' if has_10k_data else 'NO'}")
            if has_10k_data:
                print(f"    - Organizations: {node_types.get('Organization', 0):,}")
                print(f"    - Filings: {node_types.get('Filing', 0):,}")
                print(f"    - Facts: {node_types.get('Fact', 0):,}")
                print(f"    - Concepts: {node_types.get('Concept', 0):,}")
            
            # Check for news data
            has_news_data = "Article" in node_types
            print(f"  ✓ News Articles: {'YES' if has_news_data else 'NO'}")
            if has_news_data:
                print(f"    - Articles: {node_types.get('Article', 0):,}")
                print(f"    - Publishers: {node_types.get('Publisher', 0):,}")
                print(f"    - Sentiment: {node_types.get('Sentiment', 0):,}")
            
            # Check for sentiment relationships
            has_sentiment_rels = "HAS_SENTIMENT" in relationship_types
            print(f"  ✓ Sentiment Analysis: {'YES' if has_sentiment_rels else 'NO'}")
            
            # Check for news-organization linking
            has_news_org_rels = "MENTIONS_ORGANIZATION" in relationship_types
            print(f"  ✓ News-Organization Links: {'YES' if has_news_org_rels else 'NO'}")
            
            # Check for temporal relationships
            has_temporal_rels = "PRECEDES" in relationship_types
            print(f"  ✓ Temporal Relationships: {'YES' if temporal_rels else 'NO'}")
            
            print()
            
            # 4. Sample data check
            print("4. SAMPLE DATA:")
            print("-" * 15)
            
            # Sample organizations
            result = session.run("MATCH (o:Organization) RETURN o.name, o.cik LIMIT 5")
            orgs = list(result)
            if orgs:
                print("  Organizations:")
                for record in orgs:
                    print(f"    - {record['o.name']} (CIK: {record['o.cik']})")
            else:
                print("  No organizations found")
            
            # Sample articles (if any)
            if has_news_data:
                result = session.run("MATCH (a:Article) RETURN a.title, a.publishedDate LIMIT 3")
                articles = list(result)
                if articles:
                    print("  Recent Articles:")
                    for record in articles:
                        title = record['a.title'][:60] + "..." if len(record['a.title']) > 60 else record['a.title']
                        print(f"    - {title} ({record['a.publishedDate']})")
                else:
                    print("  No articles found")
            
            print()
            
            # 5. Recommendations
            print("5. RECOMMENDATIONS:")
            print("-" * 20)
            
            if not has_10k_data:
                print("  ❌ MISSING: 10-K financial data")
                print("     → Run: python ingest_10k_to_neo4j.py /path/to/10k/data")
            
            if not has_news_data:
                print("  ❌ MISSING: News articles with sentiment analysis")
                print("     → Run: python ingest_news_to_neo4j.py")
            
            if not has_sentiment_rels:
                print("  ❌ MISSING: Sentiment analysis relationships")
                print("     → Ensure news ingestion completed successfully")
            
            if not has_news_org_rels:
                print("  ❌ MISSING: News-organization linking")
                print("     → Check news ingestion pipeline")
            
            if not has_temporal_rels:
                print("  ⚠️  MISSING: Temporal relationships (PRECEDES)")
                print("     → These are optional but improve temporal analysis")
            
            if has_10k_data and has_news_data and has_sentiment_rels and has_news_org_rels:
                print("  ✅ READY: All required data is available for evaluation")
                print("     → You can run: python run_evaluation.py")
            else:
                print("  ❌ NOT READY: Missing required data for evaluation")
                print("     → Please complete data ingestion first")
            
            print()
            
            # 6. Quick fixes
            print("6. QUICK FIXES:")
            print("-" * 15)
            
            if not has_10k_data:
                print("  For 10-K data:")
                print("    python ingest_10k_to_neo4j.py 10k/INTERNATIONAL_BUSINESS_MACHINES_CORP/")
            
            if not has_news_data:
                print("  For news data:")
                print("    python ingest_news_to_neo4j.py")
            
            print()
            print("After running the ingestion scripts, run this check again to verify.")
            
    except Exception as e:
        print(f"Error checking database: {e}")
        return False
    
    finally:
        driver.close()
    
    return True


if __name__ == "__main__":
    check_database_status()
