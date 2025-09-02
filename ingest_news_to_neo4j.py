#!/usr/bin/env python3

"""
ingest_news_to_neo4j.py

Ingest news articles from RSS feeds and news APIs into a Neo4j knowledge graph,
integrating with existing 10-K filing data.

Data Sources:
- IBM Official RSS Feeds (https://www.ibm.com/ibm/syndication/us/en/)
- Google News RSS (https://news.google.com/rss/search?q=IBM+stock)
- Configurable additional RSS feeds

Features:
- Article deduplication
- Named entity recognition and linking
- Sentiment analysis
- Cross-referencing with financial concepts from 10-K filings
- Timeline analysis capabilities
- Date range filtering

Run:
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=neo4j_password
    
    python ingest_news_to_neo4j.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
"""

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
import logging

import feedparser  # pip install feedparser
import requests
from bs4 import BeautifulSoup  # type: ignore
from neo4j import GraphDatabase  # type: ignore
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    logger.info("Downloading NLTK VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)

# Try to import spacy for NER, make it optional
try:
    import spacy
    HAS_SPACY = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        HAS_SPACY = False
        nlp = None
except ImportError:
    logger.warning("spaCy not installed. Install with: pip install spacy. NER features will be disabled.")
    HAS_SPACY = False
    nlp = None

# ------------------------------
# Configuration Constants
# ------------------------------

# Default date range (last 30 days if not specified)
DEFAULT_START_DATE = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')

# RSS Feed Sources
RSS_FEEDS = {
    'ibm_official': {
        'url': 'https://www.ibm.com/blogs/think/feed/',
        'name': 'IBM Think Blog',
        'type': 'official'
    },
    'google_news_ibm': {
        'url': 'https://news.google.com/rss/search?q=IBM+stock',
        'name': 'Google News - IBM Stock',
        'type': 'news_aggregator'
    },
    'google_news_ibm_general': {
        'url': 'https://news.google.com/rss/search?q=IBM+company',
        'name': 'Google News - IBM Company',
        'type': 'news_aggregator'
    }
}

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 3

# Content filtering
MIN_ARTICLE_LENGTH = 100  # minimum characters
MAX_ARTICLE_LENGTH = 50000  # maximum characters

# ------------------------------
# Data Classes
# ------------------------------

@dataclass
class NewsArticle:
    """Represents a news article with all its metadata."""
    url: str
    title: str
    content: str
    summary: str
    published_date: datetime
    author: Optional[str]
    publisher: str
    source_feed: str
    language: str = 'en'
    
    @property
    def id(self) -> str:
        """Generate a stable ID for this article."""
        return stable_hash(self.url, self.title, self.published_date.isoformat())

@dataclass
class EntityMention:
    """Represents an entity mentioned in an article."""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int

@dataclass
class SentimentScore:
    """Represents sentiment analysis results."""
    compound: float
    positive: float
    neutral: float
    negative: float
    
    @property
    def label(self) -> str:
        """Return sentiment label based on compound score."""
        if self.compound >= 0.05:
            return 'positive'
        elif self.compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'

# ------------------------------
# Utility Functions
# ------------------------------

def stable_hash(*parts: str, maxlen: int = 40) -> str:
    """Generate a stable hash from multiple parts."""
    h = hashlib.md5("|".join([str(p) if p is not None else "" for p in parts]).encode("utf-8")).hexdigest()
    return h[:maxlen]

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common HTML entities that might slip through
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    
    return text

def is_relevant_to_ibm(text: str) -> bool:
    """Check if text content is relevant to IBM."""
    ibm_keywords = [
        'ibm', 'international business machines', 'red hat', 'watson', 'hybrid cloud',
        'artificial intelligence', 'quantum computing', 'mainframe', 'z systems',
        'power systems', 'think conference', 'arvind krishna', 'ginni rometty'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ibm_keywords)

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse various date formats into datetime object."""
    if not date_str:
        return None
    
    # Common date formats
    formats = [
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S%z',
        '%a, %d %b %Y %H:%M:%S %Z',
        '%a, %d %b %Y %H:%M:%S %z',
        '%d %b %Y %H:%M:%S %Z',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None

# ------------------------------
# Neo4j Schema Extension
# ------------------------------

NEWS_SCHEMA_QUERIES = [
    # News-specific constraints
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.url IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (pub:Publisher) REQUIRE pub.name IS UNIQUE", 
    "CREATE CONSTRAINT IF NOT EXISTS FOR (au:Author) REQUIRE au.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (feed:Feed) REQUIRE feed.url IS UNIQUE",
    
    # Sentiment analysis
    "CREATE CONSTRAINT IF NOT EXISTS FOR (sent:Sentiment) REQUIRE (sent.articleUrl, sent.type) IS NODE KEY",
    
    # Full-text search indexes
    "CREATE FULLTEXT INDEX articleContent IF NOT EXISTS FOR (a:Article) ON EACH [a.title, a.content, a.summary]",
    "CREATE FULLTEXT INDEX articleTitle IF NOT EXISTS FOR (a:Article) ON EACH [a.title]",
    
    # Date-based indexes for timeline queries
    "CREATE INDEX articlePublishedDate IF NOT EXISTS FOR (a:Article) ON (a.publishedDate)",
    "CREATE INDEX filingFiledAt IF NOT EXISTS FOR (f:Filing) ON (f.filedAt)",
]

def apply_news_schema(driver):
    """Apply news-specific schema to Neo4j database."""
    with driver.session() as session:
        for query in NEWS_SCHEMA_QUERIES:
            try:
                session.run(query)
                logger.debug(f"Applied schema: {query}")
            except Exception as e:
                logger.warning(f"Failed to apply schema query: {query}. Error: {e}")

# ------------------------------
# RSS Feed Processing
# ------------------------------

class RSSFeedProcessor:
    """Processes RSS feeds and extracts article information."""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0; +http://example.com/bot)'
        })
        
    def fetch_feed(self, feed_url: str, feed_name: str) -> List[NewsArticle]:
        """Fetch and parse RSS feed."""
        articles = []
        
        try:
            logger.info(f"Fetching RSS feed: {feed_name} ({feed_url})")
            
            # Add delay to respect rate limits
            time.sleep(REQUEST_DELAY)
            
            response = self.session.get(feed_url, timeout=30)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"Feed parsing warning for {feed_name}: {feed.bozo_exception}")
            
            logger.info(f"Found {len(feed.entries)} entries in {feed_name}")
            
            for entry in feed.entries:
                article = self._parse_feed_entry(entry, feed_name, feed_url)
                if article and self._is_in_date_range(article.published_date):
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Error fetching feed {feed_name}: {e}")
            
        return articles
    
    def _parse_feed_entry(self, entry, feed_name: str, feed_url: str) -> Optional[NewsArticle]:
        """Parse individual RSS feed entry."""
        try:
            # Extract basic information
            url = entry.get('link', '')
            title = clean_text(entry.get('title', ''))
            
            if not url or not title:
                return None
            
            # Parse publication date
            published_str = entry.get('published', '') or entry.get('updated', '')
            published_date = parse_date(published_str)
            if not published_date:
                published_date = datetime.now()
            
            # Extract content
            content = self._extract_content(entry)
            summary = clean_text(entry.get('summary', ''))
            
            # Skip if content is too short or not relevant
            if len(content) < MIN_ARTICLE_LENGTH:
                return None
                
            if not is_relevant_to_ibm(f"{title} {content} {summary}"):
                return None
            
            # Extract author and publisher
            author = self._extract_author(entry)
            publisher = self._extract_publisher(entry, feed_name)
            
            return NewsArticle(
                url=url,
                title=title,
                content=content[:MAX_ARTICLE_LENGTH],  # Truncate if too long
                summary=summary,
                published_date=published_date,
                author=author,
                publisher=publisher,
                source_feed=feed_url
            )
            
        except Exception as e:
            logger.warning(f"Error parsing feed entry: {e}")
            return None
    
    def _extract_content(self, entry) -> str:
        """Extract article content from feed entry."""
        # Try different content fields
        content_fields = ['content', 'description', 'summary']
        
        for field in content_fields:
            if field in entry:
                content_data = entry[field]
                
                if isinstance(content_data, list) and content_data:
                    content = content_data[0].get('value', '')
                elif isinstance(content_data, str):
                    content = content_data
                else:
                    continue
                
                # Clean HTML tags
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    return clean_text(soup.get_text())
        
        return ""
    
    def _extract_author(self, entry) -> Optional[str]:
        """Extract author from feed entry."""
        author_fields = ['author', 'dc:creator', 'author_detail']
        
        for field in author_fields:
            if field in entry:
                author_data = entry[field]
                if isinstance(author_data, str):
                    return clean_text(author_data)
                elif isinstance(author_data, dict) and 'name' in author_data:
                    return clean_text(author_data['name'])
        
        return None
    
    def _extract_publisher(self, entry, feed_name: str) -> str:
        """Extract publisher from feed entry or use feed name."""
        # Try to extract from entry first
        if 'source' in entry and isinstance(entry['source'], dict):
            return clean_text(entry['source'].get('title', feed_name))
        
        # Extract from feed name
        if 'Google News' in feed_name:
            return 'Google News'
        elif 'IBM' in feed_name:
            return 'IBM'
        
        return feed_name
    
    def _is_in_date_range(self, date: datetime) -> bool:
        """Check if date is within the specified range."""
        return self.start_date <= date <= self.end_date

# ------------------------------
# Content Analysis
# ------------------------------

class ContentAnalyzer:
    """Analyzes article content for entities, sentiment, and topics."""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def analyze_article(self, article: NewsArticle) -> Dict[str, Any]:
        """Perform comprehensive analysis of article content."""
        full_text = f"{article.title} {article.content}"
        
        analysis = {
            'entities': self.extract_entities(full_text),
            'sentiment': self.analyze_sentiment(full_text),
            'topics': self.extract_topics(full_text),
            'financial_concepts': self.identify_financial_concepts(full_text)
        }
        
        return analysis
    
    def extract_entities(self, text: str) -> List[EntityMention]:
        """Extract named entities from text."""
        entities = []
        
        if not HAS_SPACY or not nlp:
            return entities
        
        try:
            # Process text with spaCy
            doc = nlp(text[:5000])  # Limit length for performance
            
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE", "MONEY", "PERCENT", "DATE", "PRODUCT", "LAW"]:
                    entities.append(EntityMention(
                        text=ent.text.strip(),
                        entity_type=ent.label_,
                        confidence=1.0,  # spaCy doesn't provide confidence scores directly
                        start_pos=ent.start_char,
                        end_pos=ent.end_char
                    ))
                    
        except Exception as e:
            logger.warning(f"Error in entity extraction: {e}")
        
        return entities
    
    def analyze_sentiment(self, text: str) -> SentimentScore:
        """Analyze sentiment of text using VADER."""
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return SentimentScore(
                compound=scores['compound'],
                positive=scores['pos'],
                neutral=scores['neu'],
                negative=scores['neg']
            )
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {e}")
            return SentimentScore(0.0, 0.0, 1.0, 0.0)  # neutral default
    
    def extract_topics(self, text: str, max_topics: int = 20) -> List[str]:
        """Extract key topics/phrases from text."""
        # Simple keyword extraction based on frequency and relevance
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'among', 'throughout', 'despite', 'towards', 'upon', 'concerning'
        }
        
        # Count word frequencies
        word_counts = defaultdict(int)
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_counts[word] += 1
        
        # Return top topics
        topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in topics[:max_topics]]
    
    def identify_financial_concepts(self, text: str) -> List[str]:
        """Identify financial concepts that might link to 10-K data."""
        financial_terms = [
            'revenue', 'income', 'profit', 'loss', 'earnings', 'ebitda', 'cash flow',
            'assets', 'liabilities', 'debt', 'equity', 'shares', 'dividend', 'stock',
            'market cap', 'valuation', 'acquisition', 'merger', 'partnership',
            'quarterly results', 'annual report', 'sec filing', 'investor',
            'guidance', 'forecast', 'outlook', 'growth', 'decline'
        ]
        
        text_lower = text.lower()
        found_concepts = []
        
        for term in financial_terms:
            if term in text_lower:
                found_concepts.append(term)
        
        return found_concepts

# ------------------------------
# Neo4j Integration
# ------------------------------

class NewsNeo4jIngester:
    """Handles ingestion of news data into Neo4j."""
    
    def __init__(self, driver):
        self.driver = driver
        
    def ingest_articles(self, articles: List[NewsArticle], analyses: List[Dict[str, Any]]):
        """Ingest articles and their analyses into Neo4j."""
        if not articles:
            logger.info("No articles to ingest")
            return
        
        logger.info(f"Ingesting {len(articles)} articles into Neo4j")
        
        with self.driver.session() as session:
            # Create articles in batches
            self._create_articles(session, articles)
            
            # Create publishers and authors
            self._create_publishers_and_authors(session, articles)
            
            # Create sentiment analysis results
            self._create_sentiment_data(session, articles, analyses)
            
            # Create entity mentions and links
            self._create_entity_data(session, articles, analyses)
            
            # Link articles to organizations
            self._link_articles_to_organizations(session, articles)
            
            # Create topic associations
            self._create_topic_associations(session, articles, analyses)
            
            # Link to financial concepts from 10-K data
            self._link_to_financial_concepts(session, articles, analyses)
            
        logger.info("News ingestion completed successfully")
    
    def _create_articles(self, session, articles: List[NewsArticle]):
        """Create Article nodes."""
        article_data = []
        
        for article in articles:
            article_data.append({
                'url': article.url,
                'id': article.id,
                'title': article.title,
                'content': article.content,
                'summary': article.summary,
                'publishedDate': article.published_date.isoformat(),
                'language': article.language,
                'sourceFeed': article.source_feed,
                'contentLength': len(article.content)
            })
        
        session.run("""
            UNWIND $articles AS article
            MERGE (a:Article {url: article.url})
            SET a.id = article.id,
                a.title = article.title,
                a.content = article.content,
                a.summary = article.summary,
                a.publishedDate = article.publishedDate,
                a.language = article.language,
                a.sourceFeed = article.sourceFeed,
                a.contentLength = article.contentLength,
                a.ingestedAt = datetime()
        """, articles=article_data)
        
        logger.info(f"Created {len(article_data)} Article nodes")
    
    def _create_publishers_and_authors(self, session, articles: List[NewsArticle]):
        """Create Publisher and Author nodes and link to articles."""
        # Publishers
        publishers = set()
        authors = set()
        
        for article in articles:
            publishers.add(article.publisher)
            if article.author:
                authors.add((article.author, article.publisher))
        
        # Create publishers
        if publishers:
            publisher_data = [{'name': pub} for pub in publishers]
            session.run("""
                UNWIND $publishers AS pub
                MERGE (p:Publisher {name: pub.name})
                SET p.createdAt = coalesce(p.createdAt, datetime())
            """, publishers=publisher_data)
        
        # Create authors
        if authors:
            author_data = [{'name': auth[0], 'publisher': auth[1]} for auth in authors]
            session.run("""
                UNWIND $authors AS author
                MERGE (au:Author {name: author.name})
                SET au.createdAt = coalesce(au.createdAt, datetime())
                WITH au, author
                MATCH (p:Publisher {name: author.publisher})
                MERGE (au)-[:WRITES_FOR]->(p)
            """, authors=author_data)
        
        # Link articles to publishers and authors
        session.run("""
            MATCH (a:Article)
            MATCH (p:Publisher {name: a.sourceFeed})
            MERGE (a)-[:PUBLISHED_BY]->(p)
        """)
        
        # Link articles to authors where available
        for article in articles:
            if article.author:
                session.run("""
                    MATCH (a:Article {url: $url})
                    MATCH (au:Author {name: $author})
                    MERGE (a)-[:WRITTEN_BY]->(au)
                """, url=article.url, author=article.author)
    
    def _create_sentiment_data(self, session, articles: List[NewsArticle], analyses: List[Dict[str, Any]]):
        """Create sentiment analysis nodes."""
        sentiment_data = []
        
        for article, analysis in zip(articles, analyses):
            sentiment = analysis['sentiment']
            sentiment_data.append({
                'articleUrl': article.url,
                'type': 'overall',
                'compound': sentiment.compound,
                'positive': sentiment.positive,
                'neutral': sentiment.neutral,
                'negative': sentiment.negative,
                'label': sentiment.label
            })
        
        if sentiment_data:
            session.run("""
                UNWIND $sentiments AS sent
                MATCH (a:Article {url: sent.articleUrl})
                MERGE (s:Sentiment {articleUrl: sent.articleUrl, type: sent.type})
                SET s.compound = sent.compound,
                    s.positive = sent.positive,
                    s.neutral = sent.neutral,
                    s.negative = sent.negative,
                    s.label = sent.label,
                    s.analyzedAt = datetime()
                MERGE (a)-[:HAS_SENTIMENT]->(s)
            """, sentiments=sentiment_data)
        
        logger.info(f"Created sentiment data for {len(sentiment_data)} articles")
    
    def _create_entity_data(self, session, articles: List[NewsArticle], analyses: List[Dict[str, Any]]):
        """Create entity mentions and link to existing entities."""
        entity_links = []
        
        for article, analysis in zip(articles, analyses):
            for entity in analysis['entities']:
                entity_links.append({
                    'articleUrl': article.url,
                    'entityText': entity.text.lower(),
                    'entityType': entity.entity_type,
                    'confidence': entity.confidence
                })
        
        if entity_links:
            # Create or link to existing entities
            session.run("""
                UNWIND $links AS link
                MATCH (a:Article {url: link.articleUrl})
                MERGE (e:Entity {text: link.entityText, type: link.entityType})
                MERGE (a)-[r:MENTIONS_ENTITY]->(e)
                SET r.confidence = link.confidence,
                    r.createdAt = coalesce(r.createdAt, datetime())
            """, links=entity_links)
        
        logger.info(f"Created {len(entity_links)} entity mentions")
    
    def _link_articles_to_organizations(self, session, articles: List[NewsArticle]):
        """Link articles to existing Organization nodes."""
        # Link articles mentioning IBM to the IBM organization
        session.run("""
            MATCH (a:Article)
            WHERE toLower(a.title) CONTAINS 'ibm' OR 
                  toLower(a.content) CONTAINS 'ibm' OR
                  toLower(a.content) CONTAINS 'international business machines'
            MATCH (o:Organization)
            WHERE toLower(o.name) CONTAINS 'ibm' OR o.cik CONTAINS 'IBM'
            MERGE (a)-[:MENTIONS_ORGANIZATION]->(o)
        """)
        
        logger.info("Linked articles to organizations")
    
    def _create_topic_associations(self, session, articles: List[NewsArticle], analyses: List[Dict[str, Any]]):
        """Create topic associations for articles."""
        topic_data = []
        
        for article, analysis in zip(articles, analyses):
            for topic in analysis['topics'][:10]:  # Limit to top 10 topics
                topic_data.append({
                    'articleUrl': article.url,
                    'topicName': topic
                })
        
        if topic_data:
            session.run("""
                UNWIND $topics AS topic
                MATCH (a:Article {url: topic.articleUrl})
                MERGE (t:Topic {name: topic.topicName})
                MERGE (a)-[:DISCUSSES_TOPIC]->(t)
            """, topics=topic_data)
        
        logger.info(f"Created {len(topic_data)} topic associations")
    
    def _link_to_financial_concepts(self, session, articles: List[NewsArticle], analyses: List[Dict[str, Any]]):
        """Link articles to financial concepts from 10-K filings."""
        concept_links = []
        
        for article, analysis in zip(articles, analyses):
            for concept in analysis['financial_concepts']:
                concept_links.append({
                    'articleUrl': article.url,
                    'conceptName': concept
                })
        
        if concept_links:
            # Link to existing Concept nodes from 10-K data
            session.run("""
                UNWIND $links AS link
                MATCH (a:Article {url: link.articleUrl})
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower(link.conceptName) OR
                      toLower(link.conceptName) IN [toLower(split(c.name, '.')[-1])]
                MERGE (a)-[:DISCUSSES_FINANCIAL_CONCEPT]->(c)
            """, links=concept_links)
        
        logger.info(f"Created {len(concept_links)} financial concept links")

# ------------------------------
# Main Processing Pipeline
# ------------------------------

class NewsIngestionPipeline:
    """Main pipeline for news ingestion."""
    
    def __init__(self, start_date: str, end_date: str):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Initialize components
        self.feed_processor = RSSFeedProcessor(self.start_date, self.end_date)
        self.content_analyzer = ContentAnalyzer()
        
        # Connect to Neo4j
        uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD", "neo4j_password")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self.neo4j_ingester = NewsNeo4jIngester(self.driver)
        
        # Apply schema
        apply_news_schema(self.driver)
    
    def run(self):
        """Execute the complete news ingestion pipeline."""
        logger.info(f"Starting news ingestion for date range: {self.start_date} to {self.end_date}")
        
        all_articles = []
        all_analyses = []
        
        # Process each RSS feed
        for feed_key, feed_config in RSS_FEEDS.items():
            logger.info(f"Processing feed: {feed_config['name']}")
            
            articles = self.feed_processor.fetch_feed(
                feed_config['url'], 
                feed_config['name']
            )
            
            # Analyze content for each article
            for article in articles:
                analysis = self.content_analyzer.analyze_article(article)
                all_articles.append(article)
                all_analyses.append(analysis)
            
            logger.info(f"Processed {len(articles)} articles from {feed_config['name']}")
        
        # Deduplicate articles by URL
        unique_articles = {}
        unique_analyses = {}
        
        for article, analysis in zip(all_articles, all_analyses):
            if article.url not in unique_articles:
                unique_articles[article.url] = article
                unique_analyses[article.url] = analysis
        
        final_articles = list(unique_articles.values())
        final_analyses = list(unique_analyses.values())
        
        logger.info(f"After deduplication: {len(final_articles)} unique articles")
        
        # Ingest into Neo4j
        if final_articles:
            self.neo4j_ingester.ingest_articles(final_articles, final_analyses)
            self._log_ingestion_stats()
        else:
            logger.info("No articles found for the specified date range")
    
    def _log_ingestion_stats(self):
        """Log statistics about the ingestion."""
        with self.driver.session() as session:
            stats = session.run("""
                MATCH (a:Article)
                WHERE a.ingestedAt IS NOT NULL
                WITH a
                ORDER BY a.ingestedAt DESC
                LIMIT 1000
                OPTIONAL MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
                OPTIONAL MATCH (a)-[:MENTIONS_ENTITY]->(e:Entity)
                OPTIONAL MATCH (a)-[:DISCUSSES_TOPIC]->(t:Topic)
                RETURN 
                    count(DISTINCT a) as articles,
                    count(DISTINCT s) as sentiments,
                    count(DISTINCT e) as entities,
                    count(DISTINCT t) as topics
            """).single()
            
            logger.info(f"Ingestion stats - Articles: {stats['articles']}, "
                       f"Sentiments: {stats['sentiments']}, "
                       f"Entities: {stats['entities']}, "
                       f"Topics: {stats['topics']}")
    
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()

# ------------------------------
# CLI Interface
# ------------------------------

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Ingest news articles into Neo4j knowledge graph")
    parser.add_argument(
        '--start-date', 
        type=str, 
        default=DEFAULT_START_DATE,
        help=f'Start date for article filtering (YYYY-MM-DD). Default: {DEFAULT_START_DATE}'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        default=DEFAULT_END_DATE,
        help=f'End date for article filtering (YYYY-MM-DD). Default: {DEFAULT_END_DATE}'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
            
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return 1
    
    # Run pipeline
    pipeline = None
    try:
        pipeline = NewsIngestionPipeline(args.start_date, args.end_date)
        pipeline.run()
        logger.info("News ingestion completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
        
    finally:
        if pipeline:
            pipeline.close()

if __name__ == "__main__":
    exit(main())
