# News Integration for Knowledge Graph

This document describes the news article ingestion system that extends your existing 10-K financial data knowledge graph with real-time news analysis.

## Overview

The `ingest_news_to_neo4j.py` script integrates news articles from multiple sources into your Neo4j knowledge graph, providing:

- **Cross-referencing** between news content and financial concepts from 10-K filings
- **Timeline analysis** of news events vs financial reporting
- **Sentiment analysis** around key financial events
- **Entity linking** between news articles and existing organizational data

## Data Sources

### Currently Configured RSS Feeds

1. **IBM Official Blog** (`https://www.ibm.com/blogs/think/feed/`)
   - Official IBM corporate communications
   - Product announcements and strategic updates

2. **Google News - IBM Stock** (`https://news.google.com/rss/search?q=IBM+stock`)
   - Financial news specifically about IBM stock performance
   - Market analysis and investor sentiment

3. **Google News - IBM General** (`https://news.google.com/rss/search?q=IBM+company`)
   - Broader IBM company news
   - Industry coverage and competitive analysis

## Installation

### 1. Install Additional Dependencies

```bash
pip install -r requirements_news.txt
```

### 2. Download Required NLP Models

```bash
# For spaCy (if not already installed)
python -m spacy download en_core_web_sm

# NLTK data will be downloaded automatically on first run
```

## Usage

### Basic Usage

```bash
# Ingest news from the last 30 days (default)
python3 ingest_news_to_neo4j.py

# Specify custom date range
python3 ingest_news_to_neo4j.py --start-date 2024-01-01 --end-date 2024-01-31

# Enable debug logging
python3 ingest_news_to_neo4j.py --log-level DEBUG
```

### Environment Variables

Ensure your Neo4j connection is configured:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
```

## Neo4j Schema Extension

The script adds the following new node types and relationships to your existing schema:

### New Node Types

- **Article**: News articles with content and metadata
- **Publisher**: News sources and publications
- **Author**: Article authors
- **Sentiment**: Sentiment analysis results

### New Relationships

- `PUBLISHED_BY`: Article → Publisher
- `WRITTEN_BY`: Article → Author  
- `HAS_SENTIMENT`: Article → Sentiment
- `MENTIONS_ORGANIZATION`: Article → Organization (links to existing orgs)
- `DISCUSSES_FINANCIAL_CONCEPT`: Article → Concept (links to 10-K concepts)
- `DISCUSSES_TOPIC`: Article → Topic

## Key Features

### 1. Content Analysis Pipeline

- **Entity Recognition**: Identifies organizations, people, locations, and financial terms
- **Sentiment Analysis**: VADER sentiment scoring for overall article tone
- **Topic Extraction**: Key themes and subjects from article content
- **Financial Concept Mapping**: Links news content to 10-K filing concepts

### 2. Data Quality Controls

- **Relevance Filtering**: Only ingests articles relevant to IBM
- **Deduplication**: Prevents duplicate articles based on URL
- **Content Length Validation**: Filters out very short or excessively long content
- **Date Range Filtering**: Configurable time windows for data collection

### 3. Cross-Domain Linking

- **Organization Matching**: Links articles to existing Organization nodes
- **Concept Correlation**: Connects news discussions to financial statement concepts
- **Entity Consistency**: Maintains entity references across news and filing data

## Sample Queries

### 1. Recent IBM News with Sentiment

```cypher
MATCH (a:Article)-[:HAS_SENTIMENT]->(s:Sentiment)
MATCH (a)-[:MENTIONS_ORGANIZATION]->(o:Organization)
WHERE o.name CONTAINS 'IBM'
RETURN a.title, a.publishedDate, s.label, s.compound
ORDER BY a.publishedDate DESC
LIMIT 10
```

### 2. News Sentiment Around Earnings Announcements

```cypher
MATCH (o:Organization)-[:FILED]->(f:Filing)
MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o)
MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
WHERE f.formType = '10-K' 
  AND abs(duration.between(date(f.filedAt), date(a.publishedDate)).days) <= 30
RETURN f.filedAt, a.publishedDate, a.title, s.label, s.compound
ORDER BY f.filedAt DESC
```

### 3. Financial Concepts Discussed in News

```cypher
MATCH (a:Article)-[:DISCUSSES_FINANCIAL_CONCEPT]->(c:Concept)
MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
WHERE a.publishedDate >= date() - duration({days: 30})
RETURN c.name, 
       count(a) as articles_count,
       avg(s.compound) as avg_sentiment,
       collect(a.title)[0..3] as sample_titles
ORDER BY articles_count DESC
LIMIT 10
```

### 4. Timeline Analysis: News vs Filings

```cypher
MATCH (o:Organization {name: 'INTERNATIONAL BUSINESS MACHINES CORP'})
MATCH (o)-[:FILED]->(f:Filing)
MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o)
WHERE f.filedAt IS NOT NULL 
  AND a.publishedDate IS NOT NULL
  AND date(a.publishedDate) >= date(f.filedAt) - duration({days: 90})
  AND date(a.publishedDate) <= date(f.filedAt) + duration({days: 90})
RETURN f.filedAt as filing_date,
       f.formType,
       collect({
         date: a.publishedDate,
         title: a.title,
         sentiment: [(a)-[:HAS_SENTIMENT]->(s) | s.label][0]
       }) as related_news
ORDER BY filing_date DESC
```

## Configuration

### RSS Feed Configuration

Add new feeds by modifying the `RSS_FEEDS` dictionary in `ingest_news_to_neo4j.py`:

```python
RSS_FEEDS = {
    'your_feed_key': {
        'url': 'https://example.com/rss.xml',
        'name': 'Your Feed Name',
        'type': 'news_aggregator'  # or 'official'
    }
}
```

### Content Filtering

Adjust relevance filtering by modifying the `is_relevant_to_ibm()` function:

```python
def is_relevant_to_ibm(text: str) -> bool:
    # Add your custom keywords
    ibm_keywords = [
        'ibm', 'your_keywords_here'
    ]
    # ... rest of function
```

## Testing

Run the test script to verify everything is working:

```bash
python test_news_ingestion.py
```

This will:
- Test Neo4j connectivity
- Verify RSS feed access
- Check NLP pipeline functionality
- Display sample queries

## Performance Considerations

### Rate Limiting
- Default 1-second delay between RSS feed requests
- Configurable via `REQUEST_DELAY` constant

### Content Limits
- Articles truncated at 50,000 characters
- Minimum 100 characters for relevance
- Entity extraction limited to first 5,000 characters

### Batch Processing
- Articles processed in batches for Neo4j efficiency
- Deduplication occurs before database insertion

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install feedparser nltk requests beautifulsoup4
   ```

2. **spaCy Model Missing**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **RSS Feed Access Issues**
   - Check internet connectivity
   - Some feeds may have rate limiting
   - User-Agent string may need adjustment

4. **Neo4j Connection Issues**
   - Verify Neo4j is running
   - Check connection environment variables
   - Ensure database has sufficient memory for large datasets

### Logging

Enable debug logging for detailed troubleshooting:

```bash
python ingest_news_to_neo4j.py --log-level DEBUG
```

## Future Enhancements

Potential improvements to consider:

1. **Additional Data Sources**
   - SEC press releases API
   - Financial news APIs (Bloomberg, Reuters)
   - Social media sentiment (Twitter, Reddit)

2. **Advanced NLP**
   - Topic modeling (LDA, BERT)
   - Event extraction
   - Relationship extraction between entities

3. **Real-time Processing**
   - Streaming ingestion
   - WebSocket feeds
   - Incremental updates

4. **Enhanced Analytics**
   - Correlation analysis with stock prices
   - Predictive sentiment modeling
   - Market impact scoring

## Integration with Existing Queries

Your existing `kg_query.py` can be extended to include news data:

```python
def get_news_context(self, cik: str, days_back: int = 30):
    """Get recent news context for an organization."""
    with self.driver.session() as session:
        result = session.run("""
            MATCH (o:Organization {cik: $cik})
            MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o)
            MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
            WHERE a.publishedDate >= date() - duration({days: $days_back})
            RETURN a.title, a.publishedDate, s.label, a.url
            ORDER BY a.publishedDate DESC
            LIMIT 10
        """, cik=cik, days_back=days_back)
        return result.data()
```

This integration provides a comprehensive foundation for analyzing the relationship between news sentiment and financial performance in your knowledge graph.
