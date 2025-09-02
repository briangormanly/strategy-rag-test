#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_query.py

An MCP-capable querying tool (and CLI) that:
  1) Accepts a natural-language prompt.
  2) Retrieves the most relevant context from a local Neo4j graph.
  3) Calls a local LLM (defaults to gpt-oss-20b) with the prompt + retrieved context.
  4) Returns the model's answer and the context used.

Two modes:
  • CLI (default):   python mcp_kg_query.py --prompt "..." [--show-context]
  • MCP server:      python mcp_kg_query.py --mcp       (exposes a tool named "ask")

Environment variables (with sensible defaults):
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=neo4j_password
  NEO4J_DB=neo4j

Local model (defaults to Ollama):
  LLM_BACKEND=ollama | openai
  OLLAMA_BASE=http://localhost:11434
  OLLAMA_MODEL=gpt-oss:20b
  OPENAI_BASE= (optional) e.g., http://localhost:8000/v1  (for OpenAI-compatible servers)
  OPENAI_API_KEY= (if needed)
  OPENAI_MODEL= (e.g., llama-3.1-8b-instruct)

Requires:
  pip install neo4j requests
  (optional for MCP): pip install mcp

Notes:
  • Fulltext index "sentenceText" is used if present; otherwise falls back to substring scan.
  • Robust to any org (not IBM-specific). Tries to resolve a CIK from a ticker in the prompt.

How to run: 
    set the environment variables for the Neo4j database:
        export NEO4J_URI=bolt://localhost:7687
        export NEO4J_USER=neo4j
        export NEO4J_PASSWORD=your_password
        export NEO4J_DB=neo4j

    set the environment variables for the local model (Ollama) if not default:
        export LLM_BACKEND=ollama
        export OLLAMA_BASE=http://localhost:11434
        export OLLAMA_MODEL=gpt-oss:20b

python kg_query.py --prompt "I would like to run the following analysis on International Business Machines's (Ticker: IBM) business strategy. 1. Descriptive - Past performance - what happened and why? 2. Predictive - What will happen if the organization stays the current course? 3. Prescriptive - Things that should be considered to create a better future for the organization." --show-context
"""
import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests  # type: ignore
from neo4j import GraphDatabase  # type: ignore
from collections import defaultdict

# ---------------------------
# Utilities
# ---------------------------

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to was were will with without within into over under between during including until while than then so such via""".split())


def normspace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def stable_hash(*parts: str) -> str:
    import hashlib
    h = hashlib.md5("|".join([p if p is not None else "" for p in parts]).encode("utf-8")).hexdigest()
    return h[:40]

def pick_tokens(prompt: str, k: int = 12) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", prompt)
    toks = [w.lower() for w in words if w.lower() not in STOPWORDS and len(w) > 2]
    # keep order, dedupe
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out

# ---------------------------
# Neo4j Retrieval
# ---------------------------

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"

class Neo4jRetriever:
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

    def close(self):
        self.driver.close()

    # ---- Organization resolution (from ticker/org name) ----
    def resolve_org_from_prompt(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Try to resolve (cik, name) using uppercase ticker symbols or company substrings in the prompt.
        Returns (cik, name) or (None, None) if ambiguous.
        """
        ticker_candidates = re.findall(r"\b[A-Z]{1,5}\b", prompt)
        with self.driver.session(database=self.cfg.database) as session:
            # Try tickers first
            for sym in ticker_candidates:
                print(f"Trying ticker: {sym}")
                res = session.run(
                    """
                    MATCH (o:Organization)-[:LISTED_SECURITY]->(s:Security)
                    WHERE toUpper(s.symbol) = $sym
                    RETURN o.cik AS cik, o.name AS name LIMIT 1
                    """,
                    sym=sym,
                ).single()
                if res:
                    print(f"Resolved org: {res['name']} (CIK: {res['cik']})")
                    return res["cik"], res["name"]

            # Try org name fuzzy (first quoted phrase or longest capitalized phrase)
            m = re.search(r'"([^"]+)"', prompt)
            name_q = m.group(1) if m else None
            if not name_q:
                # naive: take longest sequence of capitalized words
                caps = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,4})\b", prompt)
                caps = sorted(caps, key=len, reverse=True)
                name_q = caps[0] if caps else None
            if name_q:
                res = session.run(
                    """
                    MATCH (o:Organization)
                    WHERE toLower(o.name) CONTAINS toLower($q)
                    RETURN o.cik AS cik, o.name AS name
                    ORDER BY size(o.name) ASC LIMIT 1
                    """,
                    q=name_q,
                ).single()
                if res:
                    return res["cik"], res["name"]
        return None, None

    # ---- Sentence fulltext (fallback aware) ----
    def top_sentences(self, query: str, cik: Optional[str], k: int = 40) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.cfg.database) as session:
            try:
                q = """
                CALL db.index.fulltext.queryNodes("sentenceText", $q) YIELD node, score
                OPTIONAL MATCH (sec:Section)-[:HAS_SENTENCE]->(node)
                OPTIONAL MATCH (f:Filing {accession: sec.filingAccession})
                OPTIONAL MATCH (o:Organization)-[:FILED]->(f)
                WITH node, score, sec, f, o
                WHERE $cik IS NULL OR o.cik = $cik
                RETURN node.id AS id, node.text AS text, score,
                       sec.item AS item, sec.title AS title,
                       f.accession AS accession, f.filedAt AS filedAt,
                       o.cik AS cik, o.name AS org
                ORDER BY score DESC
                LIMIT $k
                """
                rows = session.run(q, q=query, cik=cik, k=k).data()
                if rows:
                    return rows
            except Exception:
                pass
            # Fallback: substring scan with simple OR across tokens
            toks = pick_tokens(query, k=6)
            if not toks:
                toks = [query]
            where = " OR ".join([f"toLower(s.text) CONTAINS toLower($t{i})" for i in range(len(toks))])
            params = {f"t{i}": t for i, t in enumerate(toks)}
            params.update({"cik": cik, "k": k})
            q = f"""
            MATCH (s:Sentence)
            WHERE {where}
            OPTIONAL MATCH (sec:Section)-[:HAS_SENTENCE]->(s)
            OPTIONAL MATCH (f:Filing {{accession: sec.filingAccession}})
            OPTIONAL MATCH (o:Organization)-[:FILED]->(f)
            WITH s, sec, f, o
            WHERE $cik IS NULL OR o.cik = $cik
            RETURN s.id AS id, s.text AS text,
                   0.5 AS score,
                   sec.item AS item, sec.title AS title,
                   f.accession AS accession, f.filedAt AS filedAt,
                   o.cik AS cik, o.name AS org
            LIMIT $k
            """
            rows = session.run(q, **params).data()
            return rows

    # ---- Fact retrieval by concept-name match ----
    def top_facts(self, tokens: List[str], cik: Optional[str], limit: int = 60) -> List[Dict[str, Any]]:
        if not tokens:
            return []
        with self.driver.session(database=self.cfg.database) as session:
            q = """
            WITH $tokens AS toks
            UNWIND toks AS tok
            MATCH (c:Concept) WHERE toLower(c.name) CONTAINS toLower(tok)
            MATCH (f:Filing)-[:HAS_FACT]->(fact:Fact)-[:OF_CONCEPT]->(c)
            OPTIONAL MATCH (fact)-[:MEASURED_IN]->(u:Unit)
            OPTIONAL MATCH (fact)-[:FOR_PERIOD]->(p:Period)
            OPTIONAL MATCH (fact)-[:HAS_DIMENSION]->(d:Dimension)
            OPTIONAL MATCH (o:Organization)-[:FILED]->(f)
            WITH tok, c, fact, f, u, p, collect(DISTINCT {axis: d.axis, member: d.member}) AS dims, o
            WHERE $cik IS NULL OR o.cik = $cik
            RETURN tok, c.name AS concept, f.filedAt AS filedAt, f.accession AS accession,
                   fact.value AS value, u.measure AS unit,
                   p.start AS start, p.end AS end, p.instant AS instant,
                   dims
            ORDER BY filedAt DESC
            LIMIT $limit
            """
            rows = session.run(q, tokens=tokens, cik=cik, limit=limit).data()
            return rows

    def filing_summary(self, cik: Optional[str], limit: int = 10) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.cfg.database) as session:
            q = """
            MATCH (o:Organization)-[:FILED]->(f:Filing)
            WHERE $cik IS NULL OR o.cik = $cik
            RETURN o.cik AS cik, o.name AS org, f.accession AS accession, f.formType AS formType, f.filedAt AS filedAt
            ORDER BY f.filedAt DESC
            LIMIT $limit
            """
            return session.run(q, cik=cik, limit=limit).data()

    def get_comprehensive_context(self, prompt: str, cik: Optional[str], k: int = 50) -> Dict[str, Any]:
        """Get comprehensive context using multiple retrieval strategies including news and sentiment."""
        with self.driver.session(database=self.cfg.database) as session:
            tokens = pick_tokens(prompt, k=8)
            
            # 1. Get related facts with temporal context
            fact_context = session.run("""
                WITH $tokens AS search_tokens
                UNWIND search_tokens AS token
                MATCH (c:Concept) WHERE toLower(c.name) CONTAINS toLower(token)
                MATCH (f:Filing)-[:HAS_FACT]->(fact:Fact)-[:OF_CONCEPT]->(c)
                MATCH (fact)-[:FOR_PERIOD]->(p:Period)
                OPTIONAL MATCH (o:Organization)-[:FILED]->(f)
                OPTIONAL MATCH (fact)-[:PRECEDES*1..2]->(future_fact:Fact)
                OPTIONAL MATCH (past_fact:Fact)-[:PRECEDES*1..2]->(fact)
                WHERE $cik IS NULL OR o.cik = $cik
                RETURN c.name AS concept, fact.value AS value, 
                       CASE WHEN p.end IS NOT NULL THEN p.end 
                            WHEN p.instant IS NOT NULL THEN p.instant 
                            ELSE p.start END AS period,
                       f.filedAt AS filed, f.accession AS accession,
                       collect(DISTINCT future_fact.value)[..2] AS future_values,
                       collect(DISTINCT past_fact.value)[..2] AS past_values
                ORDER BY filed DESC
                LIMIT $k
            """, tokens=tokens, cik=cik, k=k//3).data()
            
            # 2. Get narrative with concept and entity links
            try:
                narrative_context = session.run("""
                    CALL db.index.fulltext.queryNodes("sentenceText", $query) YIELD node, score
                    MATCH (sec:Section)-[:HAS_SENTENCE]->(node)
                    MATCH (f:Filing {accession: sec.filingAccession})
                    OPTIONAL MATCH (node)-[:DISCUSSES_CONCEPT]->(c:Concept)
                    OPTIONAL MATCH (node)-[:REFERENCES_FACT]->(fact:Fact)
                    OPTIONAL MATCH (node)-[:MENTIONS_ENTITY]->(ent:Entity)
                    OPTIONAL MATCH (o:Organization)-[:FILED]->(f)
                    WHERE $cik IS NULL OR o.cik = $cik
                    RETURN node.text AS text, score, sec.item AS item, sec.title AS title,
                           f.filedAt AS filed, f.accession AS accession,
                           collect(DISTINCT c.name)[..3] AS concepts,
                           collect(DISTINCT fact.value)[..3] AS fact_values,
                           collect(DISTINCT ent.text)[..3] AS entities
                    ORDER BY score DESC
                    LIMIT $k
                """, query=prompt, cik=cik, k=k//3).data()
            except Exception:
                # Fallback without fulltext search
                narrative_context = self.top_sentences(prompt, cik, k//3)
                # Add empty arrays for missing fields
                for item in narrative_context:
                    item.setdefault('concepts', [])
                    item.setdefault('fact_values', [])
                    item.setdefault('entities', [])
            
            # 3. Get news articles with sentiment analysis
            news_context = self.get_news_context(prompt, cik, k//3)
            
            # 4. Get strategic context based on question type
            strategic_context = self.get_strategic_context(prompt, cik)
            
            return {
                "facts_with_trends": fact_context,
                "narrative_with_links": narrative_context,
                "news_with_sentiment": news_context,
                "strategic_context": strategic_context
            }

    def get_strategic_context(self, prompt: str, cik: Optional[str]) -> Dict[str, Any]:
        """Get context optimized for strategic analysis questions."""
        with self.driver.session(database=self.cfg.database) as session:
            # Identify question type
            prompt_lower = prompt.lower()
            is_trend_question = any(word in prompt_lower for word in 
                                   ["trend", "change", "growth", "decline", "over time", "historical", "compare"])
            is_risk_question = any(word in prompt_lower for word in 
                                  ["risk", "threat", "challenge", "concern", "problem", "uncertainty"])
            is_strategy_question = any(word in prompt_lower for word in 
                                     ["strategy", "strategic", "direction", "future", "plan", "initiative"])
            
            context = {}
            
            if is_trend_question:
                # Get time-series data for key financial metrics
                try:
                    context["trends"] = session.run("""
                        MATCH (o:Organization {cik: $cik})-[:FILED]->(f:Filing)-[:HAS_FACT]->(fact:Fact)
                        MATCH (fact)-[:OF_CONCEPT]->(c:Concept)
                        MATCH (fact)-[:FOR_PERIOD]->(p:Period)
                        WHERE c.name CONTAINS 'Revenue' OR c.name CONTAINS 'NetIncome' OR 
                              c.name CONTAINS 'TotalAssets' OR c.name CONTAINS 'TotalDebt' OR
                              c.name CONTAINS 'CashAndCashEquivalents'
                        RETURN c.name AS concept, fact.value AS value, 
                               CASE WHEN p.end IS NOT NULL THEN p.end 
                                    WHEN p.instant IS NOT NULL THEN p.instant 
                                    ELSE p.start END AS period, 
                               f.filedAt AS filed, f.accession AS accession
                        ORDER BY c.name, period DESC
                        LIMIT 30
                    """, cik=cik).data()
                except Exception as e:
                    context["trends"] = []
            
            if is_risk_question:
                # Get risk-related narrative
                try:
                    context["risks"] = session.run("""
                        MATCH (s:Sentence) 
                        WHERE toLower(s.text) CONTAINS 'risk' OR 
                              toLower(s.text) CONTAINS 'uncertainty' OR
                              toLower(s.text) CONTAINS 'challenge' OR
                              toLower(s.text) CONTAINS 'threat'
                        MATCH (sec:Section)-[:HAS_SENTENCE]->(s)
                        MATCH (f:Filing {accession: sec.filingAccession})
                        MATCH (o:Organization {cik: $cik})-[:FILED]->(f)
                        RETURN s.text AS text, sec.item AS item, sec.title AS title, 
                               f.filedAt AS filed, f.accession AS accession
                        ORDER BY filed DESC
                        LIMIT 15
                    """, cik=cik).data()
                except Exception:
                    context["risks"] = []
            
            if is_strategy_question:
                # Get strategy-related content
                try:
                    context["strategy"] = session.run("""
                        MATCH (s:Sentence) 
                        WHERE toLower(s.text) CONTAINS 'strategy' OR 
                              toLower(s.text) CONTAINS 'strategic' OR
                              toLower(s.text) CONTAINS 'initiative' OR
                              toLower(s.text) CONTAINS 'future' OR
                              toLower(s.text) CONTAINS 'plan'
                        MATCH (sec:Section)-[:HAS_SENTENCE]->(s)
                        MATCH (f:Filing {accession: sec.filingAccession})
                        MATCH (o:Organization {cik: $cik})-[:FILED]->(f)
                        RETURN s.text AS text, sec.item AS item, sec.title AS title, 
                               f.filedAt AS filed, f.accession AS accession
                        ORDER BY filed DESC
                        LIMIT 15
                    """, cik=cik).data()
                except Exception:
                    context["strategy"] = []
            
            return context

    def get_news_context(self, prompt: str, cik: Optional[str], limit: int = 20) -> Dict[str, Any]:
        """Get news articles with sentiment analysis for context."""
        with self.driver.session(database=self.cfg.database) as session:
            tokens = pick_tokens(prompt, k=6)
            
            # Get recent news articles with sentiment
            try:
                # First try fulltext search on articles
                news_articles = session.run("""
                    CALL db.index.fulltext.queryNodes("articleContent", $query) YIELD node, score
                    MATCH (node:Article)
                    OPTIONAL MATCH (node)-[:MENTIONS_ORGANIZATION]->(o:Organization)
                    OPTIONAL MATCH (node)-[:HAS_SENTIMENT]->(s:Sentiment)
                    OPTIONAL MATCH (node)-[:PUBLISHED_BY]->(pub:Publisher)
                    OPTIONAL MATCH (node)-[:WRITTEN_BY]->(au:Author)
                    OPTIONAL MATCH (node)-[:DISCUSSES_FINANCIAL_CONCEPT]->(c:Concept)
                    OPTIONAL MATCH (node)-[:MENTIONS_ENTITY]->(ent:Entity)
                    WHERE $cik IS NULL OR o.cik = $cik
                    RETURN node.title AS title, node.publishedDate AS published,
                           node.url AS url, node.summary AS summary,
                           s.label AS sentiment_label, s.compound AS sentiment_score,
                           s.positive AS positive, s.negative AS negative, s.neutral AS neutral,
                           pub.name AS publisher, au.name AS author,
                           collect(DISTINCT c.name)[..3] AS financial_concepts,
                           collect(DISTINCT ent.text)[..5] AS entities,
                           score, o.name AS organization
                    ORDER BY score DESC, published DESC
                    LIMIT $limit
                """, query=prompt, cik=cik, limit=limit).data()
            except Exception:
                # Fallback: keyword-based search on articles
                if tokens:
                    where_clauses = []
                    params = {"cik": cik, "limit": limit}
                    
                    for i, token in enumerate(tokens[:4]):
                        where_clauses.append(f"(toLower(a.title) CONTAINS toLower($token{i}) OR toLower(a.content) CONTAINS toLower($token{i}))")
                        params[f"token{i}"] = token
                    
                    where_condition = " OR ".join(where_clauses) if where_clauses else "true"
                    
                    news_articles = session.run(f"""
                        MATCH (a:Article)
                        WHERE {where_condition}
                        OPTIONAL MATCH (a)-[:MENTIONS_ORGANIZATION]->(o:Organization)
                        OPTIONAL MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
                        OPTIONAL MATCH (a)-[:PUBLISHED_BY]->(pub:Publisher)
                        OPTIONAL MATCH (a)-[:WRITTEN_BY]->(au:Author)
                        OPTIONAL MATCH (a)-[:DISCUSSES_FINANCIAL_CONCEPT]->(c:Concept)
                        OPTIONAL MATCH (a)-[:MENTIONS_ENTITY]->(ent:Entity)
                        WHERE $cik IS NULL OR o.cik = $cik
                        RETURN a.title AS title, a.publishedDate AS published,
                               a.url AS url, a.summary AS summary,
                               s.label AS sentiment_label, s.compound AS sentiment_score,
                               s.positive AS positive, s.negative AS negative, s.neutral AS neutral,
                               pub.name AS publisher, au.name AS author,
                               collect(DISTINCT c.name)[..3] AS financial_concepts,
                               collect(DISTINCT ent.text)[..5] AS entities,
                               0.5 AS score, o.name AS organization
                        ORDER BY published DESC
                        LIMIT $limit
                    """, **params).data()
                else:
                    news_articles = []
            
            # Get news sentiment timeline around filing dates
            sentiment_timeline = []
            if cik:
                try:
                    sentiment_timeline = session.run("""
                        MATCH (o:Organization {cik: $cik})-[:FILED]->(f:Filing)
                        MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o)
                        MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
                        WHERE f.filedAt IS NOT NULL AND a.publishedDate IS NOT NULL
                          AND abs(duration.between(date(f.filedAt), date(a.publishedDate)).days) <= 60
                        RETURN f.filedAt AS filing_date, f.formType AS form_type, f.accession AS accession,
                               a.publishedDate AS news_date, a.title AS news_title,
                               s.label AS sentiment, s.compound AS sentiment_score,
                               duration.between(date(f.filedAt), date(a.publishedDate)).days AS days_from_filing
                        ORDER BY filing_date DESC, news_date DESC
                        LIMIT 15
                    """, cik=cik).data()
                except Exception:
                    sentiment_timeline = []
            
            # Get sentiment summary for recent period
            sentiment_summary = {}
            if cik:
                try:
                    sentiment_summary = session.run("""
                        MATCH (o:Organization {cik: $cik})
                        MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o)
                        MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
                        WHERE a.publishedDate >= date() - duration({days: 90})
                        RETURN 
                            count(a) AS total_articles,
                            avg(s.compound) AS avg_sentiment,
                            sum(CASE WHEN s.label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
                            sum(CASE WHEN s.label = 'negative' THEN 1 ELSE 0 END) AS negative_count,
                            sum(CASE WHEN s.label = 'neutral' THEN 1 ELSE 0 END) AS neutral_count
                    """, cik=cik).single()
                    
                    if sentiment_summary:
                        sentiment_summary = dict(sentiment_summary)
                    else:
                        sentiment_summary = {}
                except Exception:
                    sentiment_summary = {}
            
            return {
                "articles": news_articles,
                "sentiment_timeline": sentiment_timeline,
                "sentiment_summary": sentiment_summary
            }

# ---------------------------
# Context building
# ---------------------------

def build_enhanced_context(prompt: str, cik: Optional[str], org_name: Optional[str],
                          context_data: Dict[str, Any]) -> str:
    """Build context with temporal trends and cross-references."""
    lines = []
    lines.append("<<PROMPT>>")
    lines.append(prompt.strip())
    lines.append("")
    
    if cik or org_name:
        lines.append("<<ORGANIZATION>>")
        lines.append(f"name: {org_name or 'Unknown'}")
        lines.append(f"cik: {cik or 'Unknown'}")
        lines.append("")
    
    # Enhanced facts with trends
    facts_with_trends = context_data.get("facts_with_trends", [])
    if facts_with_trends:
        lines.append("<<FINANCIAL METRICS WITH TRENDS>>")
        by_concept = defaultdict(list)
        for fact in facts_with_trends:
            by_concept[fact["concept"]].append(fact)
        
        for concept, facts in list(by_concept.items())[:8]:
            lines.append(f"- {concept}:")
            for fact in sorted(facts, key=lambda x: x.get("period", "") or "", reverse=True)[:3]:
                trend_info = ""
                if fact.get("past_values") and any(fact["past_values"]):
                    trend_info += f" (prev: {', '.join(map(str, fact['past_values']))})"
                if fact.get("future_values") and any(fact["future_values"]):
                    trend_info += f" (next: {', '.join(map(str, fact['future_values']))})"
                
                lines.append(f"    • {fact.get('value')} @ {fact.get('period', '?')} (filed {fact.get('filed','?')} acc {fact.get('accession','?')}){trend_info}")
        lines.append("")
    
    # Enhanced narrative with concept links
    narrative_with_links = context_data.get("narrative_with_links", [])
    if narrative_with_links:
        lines.append("<<NARRATIVE WITH FINANCIAL CONCEPT LINKS>>")
        for item in narrative_with_links[:15]:
            concepts_str = f" [Concepts: {', '.join(item.get('concepts', []))}]" if item.get('concepts') else ""
            facts_str = f" [Values: {', '.join(map(str, item.get('fact_values', [])))}]" if item.get('fact_values') else ""
            entities_str = f" [Entities: {', '.join(item.get('entities', []))}]" if item.get('entities') else ""
            
            text_preview = normspace(item.get('text', ''))[:200]
            lines.append(f"- [Item {item.get('item', '?')}] {item.get('filed', '?')}: {text_preview}...{concepts_str}{facts_str}{entities_str}")
        lines.append("")
    
    # News and sentiment context
    news_context = context_data.get("news_with_sentiment", {})
    
    if news_context.get("sentiment_summary") and any(news_context["sentiment_summary"].values()):
        lines.append("<<NEWS SENTIMENT SUMMARY (Last 90 Days)>>")
        summary = news_context["sentiment_summary"]
        total = summary.get("total_articles", 0)
        if total > 0:
            avg_sentiment = summary.get("avg_sentiment", 0)
            pos_count = summary.get("positive_count", 0)
            neg_count = summary.get("negative_count", 0)
            neu_count = summary.get("neutral_count", 0)
            
            lines.append(f"- Total articles: {total}")
            lines.append(f"- Average sentiment: {avg_sentiment:.3f} ({'Positive' if avg_sentiment > 0.05 else 'Negative' if avg_sentiment < -0.05 else 'Neutral'})")
            lines.append(f"- Distribution: {pos_count} positive, {neg_count} negative, {neu_count} neutral")
        lines.append("")
    
    if news_context.get("articles"):
        lines.append("<<RECENT NEWS ARTICLES WITH SENTIMENT>>")
        for article in news_context["articles"][:12]:
            sentiment_info = ""
            if article.get("sentiment_label") and article.get("sentiment_score") is not None:
                sentiment_info = f" [Sentiment: {article['sentiment_label']} ({article['sentiment_score']:.2f})]"
            
            concepts_info = ""
            if article.get("financial_concepts"):
                concepts_info = f" [Financial: {', '.join(article['financial_concepts'])}]"
            
            entities_info = ""
            if article.get("entities"):
                entities_info = f" [Entities: {', '.join(article['entities'][:3])}]"
            
            publisher_info = f" [{article.get('publisher', 'Unknown')}]" if article.get('publisher') else ""
            
            title = article.get('title', 'No title')[:100]
            published = article.get('published', 'Unknown date')
            
            lines.append(f"- {published}{publisher_info}: {title}...{sentiment_info}{concepts_info}{entities_info}")
        lines.append("")
    
    if news_context.get("sentiment_timeline"):
        lines.append("<<NEWS SENTIMENT AROUND FILING DATES>>")
        for timeline_item in news_context["sentiment_timeline"][:10]:
            filing_date = timeline_item.get("filing_date", "?")
            form_type = timeline_item.get("form_type", "?")
            news_date = timeline_item.get("news_date", "?")
            news_title = timeline_item.get("news_title", "No title")[:80]
            sentiment = timeline_item.get("sentiment", "?")
            sentiment_score = timeline_item.get("sentiment_score", 0)
            days_diff = timeline_item.get("days_from_filing", 0)
            
            timing = f"{abs(days_diff)} days {'after' if days_diff > 0 else 'before'}" if days_diff != 0 else "same day"
            
            lines.append(f"- {form_type} filing {filing_date} → {news_date} ({timing}): {news_title}... [Sentiment: {sentiment} ({sentiment_score:.2f})]")
        lines.append("")
    
    # Strategic context sections
    strategic_context = context_data.get("strategic_context", {})
    
    if strategic_context.get("trends"):
        lines.append("<<TIME-SERIES FINANCIAL DATA>>")
        trends_by_concept = defaultdict(list)
        for trend in strategic_context["trends"]:
            trends_by_concept[trend["concept"]].append(trend)
        
        for concept, trend_data in list(trends_by_concept.items())[:5]:
            lines.append(f"- {concept}:")
            for data in sorted(trend_data, key=lambda x: x.get("period", "") or "", reverse=True)[:4]:
                lines.append(f"    • {data.get('value')} @ {data.get('period', '?')} (acc {data.get('accession', '?')})")
        lines.append("")
    
    if strategic_context.get("risks"):
        lines.append("<<RISK FACTORS & CHALLENGES>>")
        for risk in strategic_context["risks"][:8]:
            text_preview = normspace(risk.get('text', ''))[:150]
            lines.append(f"- [Item {risk.get('item', '?')}] {risk.get('filed', '?')}: {text_preview}...")
        lines.append("")
    
    if strategic_context.get("strategy"):
        lines.append("<<STRATEGIC INITIATIVES & DIRECTION>>")
        for strategy in strategic_context["strategy"][:8]:
            text_preview = normspace(strategy.get('text', ''))[:150]
            lines.append(f"- [Item {strategy.get('item', '?')}] {strategy.get('filed', '?')}: {text_preview}...")
        lines.append("")
    
    # Final guide to model
    lines.append("<<INSTRUCTIONS>>")
    lines.append(
        "Using the enhanced context above, provide comprehensive strategic analysis. "
        "Connect quantitative financial trends with qualitative narrative insights and market sentiment. "
        "Reference specific accession numbers, time periods, news sentiment scores, and cross-link related concepts. "
        "For trend analysis, explain the progression and implications over time, including market perception. "
        "Correlate news sentiment patterns with financial events and filing dates to identify market reactions. "
        "Use sentiment timeline data to understand how the market responded to financial disclosures. "
        "Organize your response in clear sections addressing the user's specific questions with integrated analysis."
    )
    return "\n".join(lines)

# Legacy function for backward compatibility
def build_context(prompt: str, cik: Optional[str], org_name: Optional[str],
                  sentences: List[Dict[str, Any]], facts: List[Dict[str, Any]], filings: List[Dict[str, Any]]) -> str:
    """Legacy context building function - maintained for compatibility."""
    context_data = {
        "facts_with_trends": [{"concept": f.get("concept"), "value": f.get("value"), 
                              "period": f.get("instant") or f.get("end") or f.get("start"),
                              "filed": f.get("filedAt"), "accession": f.get("accession")} for f in facts],
        "narrative_with_links": [{"text": s.get("text"), "item": s.get("item"), 
                                 "filed": s.get("filedAt"), "accession": s.get("accession"),
                                 "concepts": [], "fact_values": [], "entities": []} for s in sentences],
        "strategic_context": {}
    }
    return build_enhanced_context(prompt, cik, org_name, context_data)

# ---------------------------
# Local LLM adapters
# ---------------------------

class LLMBase:
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError

class OllamaLLM(LLMBase):
    def __init__(self, base: str, model: str, timeout: int = 120):
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        url = f"{self.base}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # ollama responses may be in 'message' or 'messages'
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")
        if "messages" in data and isinstance(data["messages"], list) and data["messages"]:
            return data["messages"][-1].get("content", "")
        # fallback to /api/generate if /api/chat not supported
        url = f"{self.base}/api/generate"
        payload = {
            "model": self.model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "")

class OpenAICompatLLM(LLMBase):
    def __init__(self, base: str, model: str, api_key: Optional[str] = None, timeout: int = 120):
        self.base = base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        url = f"{self.base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

def make_llm_from_env() -> LLMBase:
    backend = os.getenv("LLM_BACKEND", "ollama").lower().strip()
    if backend == "openai":
        base = os.getenv("OPENAI_BASE", "http://localhost:8000/v1")
        model = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instruct")
        key = os.getenv("OPENAI_API_KEY")
        return OpenAICompatLLM(base, model, key)
    # default: ollama
    base = os.getenv("OLLAMA_BASE", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    return OllamaLLM(base, model)

# ---------------------------
# Query pipeline
# ---------------------------

def run_pipeline(prompt: str, cfg: Neo4jConfig, show_context: bool = False) -> Dict[str, Any]:
    retriever = Neo4jRetriever(cfg)
    try:
        # Resolve organization
        cik, org_name = retriever.resolve_org_from_prompt(prompt)
        
        # Get comprehensive context using enhanced retrieval
        context_data = retriever.get_comprehensive_context(prompt, cik=cik, k=50)
        
        # Build enhanced context
        ctx = build_enhanced_context(prompt, cik, org_name, context_data)

        # Generate response with enhanced system prompt
        llm = make_llm_from_env()
        system = ("You are an expert financial analyst and strategic planning assistant. "
                 "You have access to comprehensive financial data, narrative disclosures, temporal trends, "
                 "news articles with sentiment analysis, and cross-referenced market intelligence. "
                 "Provide strategic insights that connect quantitative metrics with qualitative context and market sentiment. "
                 "Structure your responses clearly with specific sections addressing different aspects of the analysis. "
                 "Always cite specific data points, accession numbers, time periods, and news sentiment when making claims. "
                 "When analyzing trends, explain both the numerical progression, business implications, and market perception. "
                 "Correlate news sentiment with financial performance and filing dates to provide comprehensive analysis.")
        
        answer = llm.generate(system, ctx)

        if show_context:
            return {"answer": answer, "context": ctx, "cik": cik, "org": org_name, "context_data": context_data}
        return {"answer": answer, "cik": cik, "org": org_name}
    finally:
        retriever.close()

# ---------------------------
# CLI
# ---------------------------

def cli_main(args: argparse.Namespace) -> int:
    cfg = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        database=os.getenv("NEO4J_DB", "neo4j"),
    )
    if not args.prompt:
        print("Enter your prompt (end with Ctrl-D):")
        prompt = sys.stdin.read().strip()
    else:
        prompt = args.prompt.strip()

    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return 2

    result = run_pipeline(prompt, cfg, show_context=args.show_context)
    print("\n=== ANSWER ===\n")
    print(result["answer"].strip())
    if args.show_context and "context" in result:
        print("\n=== CONTEXT USED ===\n")
        print(result["context"])
    return 0

# ---------------------------
# MCP server (optional)
# ---------------------------

def mcp_main() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as e:
        print("The 'mcp' package is required for --mcp mode. Try: pip install mcp", file=sys.stderr)
        return 2

    app = FastMCP("strategery-kg")

    @app.tool()
    def ask(prompt: str, show_context: bool = False) -> dict:
        """
        Query the Strategery knowledge graph and a local model.

        Args:
            prompt: The user's question.
            show_context: If true, returns the retrieved context along with the answer.
        Returns: dict with 'answer' and optionally 'context'.
        """
        cfg = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
            database=os.getenv("NEO4J_DB", "neo4j"),
        )
        return run_pipeline(prompt, cfg, show_context=show_context)

    app.run()
    return 0

# ---------------------------
# Entrypoint
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MCP/CLI Querying Tool for Strategery Neo4j Graph")
    ap.add_argument("--prompt", type=str, help="Natural-language question to ask")
    ap.add_argument("--show-context", action="store_true", help="Print the retrieved context")
    ap.add_argument("--mcp", action="store_true", help="Run as an MCP tool server instead of CLI")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.mcp:
        sys.exit(mcp_main())
    else:
        sys.exit(cli_main(args))

if __name__ == "__main__":
    main()