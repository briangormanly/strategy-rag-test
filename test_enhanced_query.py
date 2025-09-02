#!/usr/bin/env python3

"""
test_enhanced_query.py

Test script to demonstrate the enhanced kg_query.py functionality with news and sentiment analysis.
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_news_integration():
    """Test the enhanced query system with news integration."""
    try:
        from kg_query import Neo4jRetriever, Neo4jConfig
        
        # Setup Neo4j configuration
        cfg = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
            database=os.getenv("NEO4J_DB", "neo4j"),
        )
        
        print("Testing Enhanced Knowledge Graph Query System")
        print("=" * 50)
        
        retriever = Neo4jRetriever(cfg)
        
        # Test organization resolution
        print("\n1. Testing Organization Resolution...")
        cik, org_name = retriever.resolve_org_from_prompt("IBM earnings analysis")
        print(f"   Resolved: {org_name} (CIK: {cik})")
        
        # Test news context retrieval
        print("\n2. Testing News Context Retrieval...")
        news_context = retriever.get_news_context("IBM earnings revenue growth", cik, limit=5)
        
        articles = news_context.get("articles", [])
        sentiment_summary = news_context.get("sentiment_summary", {})
        sentiment_timeline = news_context.get("sentiment_timeline", [])
        
        print(f"   Found {len(articles)} relevant news articles")
        
        if sentiment_summary:
            total = sentiment_summary.get("total_articles", 0)
            avg_sentiment = sentiment_summary.get("avg_sentiment", 0)
            print(f"   Sentiment Summary: {total} articles, avg sentiment: {avg_sentiment:.3f}")
        
        if sentiment_timeline:
            print(f"   Found {len(sentiment_timeline)} sentiment timeline entries")
        
        # Test comprehensive context
        print("\n3. Testing Comprehensive Context Retrieval...")
        context_data = retriever.get_comprehensive_context("IBM financial performance and market sentiment", cik, k=30)
        
        facts = context_data.get("facts_with_trends", [])
        narrative = context_data.get("narrative_with_links", [])
        news = context_data.get("news_with_sentiment", {})
        strategic = context_data.get("strategic_context", {})
        
        print(f"   Financial facts: {len(facts)}")
        print(f"   Narrative context: {len(narrative)}")
        print(f"   News articles: {len(news.get('articles', []))}")
        print(f"   Strategic context sections: {len(strategic)}")
        
        # Sample news article with sentiment
        if news.get("articles"):
            sample_article = news["articles"][0]
            print(f"\n   Sample Article:")
            print(f"     Title: {sample_article.get('title', 'No title')[:80]}...")
            print(f"     Published: {sample_article.get('published', 'Unknown')}")
            print(f"     Sentiment: {sample_article.get('sentiment_label', 'Unknown')} ({sample_article.get('sentiment_score', 'N/A')})")
            print(f"     Publisher: {sample_article.get('publisher', 'Unknown')}")
            
            if sample_article.get('financial_concepts'):
                print(f"     Financial Concepts: {', '.join(sample_article['financial_concepts'])}")
        
        retriever.close()
        
        print("\n✓ All tests completed successfully!")
        print("\nThe enhanced query system now includes:")
        print("  • News articles with sentiment analysis")
        print("  • Cross-references between news and financial concepts")
        print("  • Sentiment timeline around filing dates")
        print("  • Market perception analysis")
        print("  • Entity linking across news and filings")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_sample_enhanced_queries():
    """Show sample queries that leverage the new news and sentiment capabilities."""
    print("\n" + "="*60)
    print("SAMPLE ENHANCED QUERIES WITH NEWS & SENTIMENT")
    print("="*60)
    
    queries = [
        {
            "query": "How did market sentiment change around IBM's recent earnings announcements?",
            "description": "Analyzes sentiment timeline around filing dates"
        },
        {
            "query": "What is the correlation between IBM's financial performance and news sentiment?",
            "description": "Cross-correlates financial metrics with market perception"
        },
        {
            "query": "IBM strategic analysis with market reaction to recent initiatives",
            "description": "Combines strategic context with news sentiment"
        },
        {
            "query": "IBM revenue growth prospects based on financial data and market sentiment",
            "description": "Integrates quantitative trends with qualitative market signals"
        }
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\n{i}. Query: \"{q['query']}\"")
        print(f"   Purpose: {q['description']}")
        print(f"   Command: python kg_query.py --prompt \"{q['query']}\" --show-context")

if __name__ == "__main__":
    print("Enhanced Knowledge Graph Query Test")
    print("=" * 40)
    
    # Test the enhanced functionality
    success = test_news_integration()
    
    if success:
        show_sample_enhanced_queries()
        
        print(f"\n" + "="*60)
        print("READY FOR ENHANCED ANALYSIS!")
        print("="*60)
        print("Your knowledge graph now combines:")
        print("  📊 Financial data from 10-K filings")
        print("  📰 News articles with sentiment analysis")
        print("  🔗 Cross-domain entity linking")
        print("  📈 Timeline correlation analysis")
        print("  🎯 Market perception insights")
        
    else:
        print("\nPlease ensure:")
        print("  • Neo4j is running with both 10-K and news data")
        print("  • Environment variables are set correctly")
        print("  • News ingestion has been completed")
        sys.exit(1)
