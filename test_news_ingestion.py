#!/usr/bin/env python3

"""
test_news_ingestion.py

Simple test script to demonstrate news ingestion functionality.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic news ingestion functionality."""
    try:
        from ingest_news_to_neo4j import NewsIngestionPipeline
        
        # Test with last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"Testing news ingestion from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Create pipeline (this will test Neo4j connection)
        pipeline = NewsIngestionPipeline(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        print("✓ Pipeline created successfully")
        print("✓ Neo4j connection established")
        print("✓ Schema applied")
        
        # Test RSS feed processing (without full ingestion)
        print("\nTesting RSS feed access...")
        
        from ingest_news_to_neo4j import RSS_FEEDS, RSSFeedProcessor
        
        feed_processor = RSSFeedProcessor(start_date, end_date)
        
        for feed_key, feed_config in list(RSS_FEEDS.items())[:1]:  # Test just one feed
            print(f"Testing feed: {feed_config['name']}")
            articles = feed_processor.fetch_feed(feed_config['url'], feed_config['name'])
            print(f"✓ Found {len(articles)} relevant articles")
            
            if articles:
                print(f"Sample article: {articles[0].title[:100]}...")
                break
        
        pipeline.close()
        print("\n✓ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Please install: pip install -r requirements_news.txt")
        return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def show_sample_queries():
    """Show sample Neo4j queries for news analysis."""
    queries = [
        {
            "name": "Recent IBM News Articles",
            "query": """
            MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o:Organization)
            WHERE o.name CONTAINS 'IBM'
            RETURN a.title, a.publishedDate, a.url
            ORDER BY a.publishedDate DESC
            LIMIT 10
            """
        },
        {
            "name": "News Sentiment Around Earnings",
            "query": """
            MATCH (a:Article)-[:HAS_SENTIMENT]->(s:Sentiment)
            MATCH (a)-[:DISCUSSES_FINANCIAL_CONCEPT]->(c:Concept)
            WHERE toLower(a.content) CONTAINS 'earnings' OR toLower(a.title) CONTAINS 'earnings'
            RETURN a.title, s.label, s.compound, a.publishedDate
            ORDER BY a.publishedDate DESC
            """
        },
        {
            "name": "Timeline: News vs Financial Filings",
            "query": """
            MATCH (o:Organization)-[:FILED]->(f:Filing)
            MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o)
            WHERE f.filedAt IS NOT NULL AND a.publishedDate IS NOT NULL
            RETURN 
                f.filedAt as filing_date,
                a.publishedDate as news_date,
                f.formType,
                a.title,
                abs(duration.between(date(f.filedAt), date(a.publishedDate)).days) as days_apart
            ORDER BY filing_date DESC
            """
        },
        {
            "name": "Most Mentioned Entities in News",
            "query": """
            MATCH (a:Article)-[:MENTIONS_ENTITY]->(e:Entity)
            WHERE a.publishedDate >= date() - duration({days: 30})
            RETURN e.text, e.type, count(a) as mention_count
            ORDER BY mention_count DESC
            LIMIT 20
            """
        }
    ]
    
    print("\n" + "="*60)
    print("SAMPLE NEO4J QUERIES FOR NEWS ANALYSIS")
    print("="*60)
    
    for i, query_info in enumerate(queries, 1):
        print(f"\n{i}. {query_info['name']}")
        print("-" * (len(query_info['name']) + 3))
        print(query_info['query'].strip())

if __name__ == "__main__":
    print("News Ingestion Test Script")
    print("=" * 30)
    
    # Test basic functionality
    success = test_basic_functionality()
    
    if success:
        show_sample_queries()
    else:
        print("\nPlease fix the issues above before proceeding.")
        sys.exit(1)
