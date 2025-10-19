#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_framework_optimized.py

Optimized evaluation framework to handle Neo4j memory constraints.
Uses smaller, more efficient queries to avoid memory pool exhaustion.
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import re

# Import our existing components
from kg_query import Neo4jRetriever, Neo4jConfig, make_llm_from_env, build_enhanced_context
from neo4j import GraphDatabase


@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for comparing approaches."""
    
    # Accuracy & Correctness
    factual_accuracy: float = 0.0
    citation_accuracy: float = 0.0
    numerical_precision: float = 0.0
    
    # Context Quality
    context_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    
    # Response Quality
    completeness_score: float = 0.0
    depth_score: float = 0.0
    coherence_score: float = 0.0
    
    # Advanced Capabilities
    temporal_analysis_score: float = 0.0
    cross_reference_score: float = 0.0
    sentiment_integration_score: float = 0.0
    strategic_insight_score: float = 0.0
    
    # Performance
    response_time: float = 0.0
    context_size: int = 0
    token_efficiency: float = 0.0
    
    # Cost Analysis
    estimated_cost: float = 0.0
    cost_per_insight: float = 0.0


@dataclass
class TestQuery:
    """Standardized test query with expected outcomes."""
    id: str
    query: str
    category: str  # factual, analytical, strategic, temporal, cross_domain
    expected_elements: List[str]  # Key elements that should appear in response
    expected_citations: List[str]  # Expected citation types
    difficulty: str  # easy, medium, hard
    description: str


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query_id: str
    approach: str  # "graphrag" or "direct_llm"
    response: str
    context_used: str
    metrics: EvaluationMetrics
    timestamp: datetime
    raw_data: Dict[str, Any]


class OptimizedNeo4jRetriever:
    """Memory-optimized Neo4j retriever with smaller queries."""
    
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    
    def close(self):
        self.driver.close()
    
    def get_optimized_context(self, prompt: str, cik: str, k: int = 20) -> Dict[str, Any]:
        """Get optimized context with smaller, more efficient queries."""
        with self.driver.session(database=self.cfg.database) as session:
            # 1. Get basic facts (simplified query)
            facts = session.run("""
                MATCH (o:Organization {cik: $cik})-[:FILED]->(f:Filing)-[:HAS_FACT]->(fact:Fact)
                MATCH (fact)-[:OF_CONCEPT]->(c:Concept)
                MATCH (fact)-[:FOR_PERIOD]->(p:Period)
                WHERE c.name CONTAINS 'Revenue' OR c.name CONTAINS 'NetIncome' OR 
                      c.name CONTAINS 'TotalAssets' OR c.name CONTAINS 'Cash'
                RETURN c.name AS concept, fact.value AS value, 
                       CASE WHEN p.end IS NOT NULL THEN p.end 
                            WHEN p.instant IS NOT NULL THEN p.instant 
                            ELSE p.start END AS period,
                       f.filedAt AS filed, f.accession AS accession
                ORDER BY f.filedAt DESC
                LIMIT $k
            """, cik=cik, k=k//2).data()
            
            # 2. Get recent news (simplified)
            news = []
            try:
                news = session.run("""
                    MATCH (o:Organization {cik: $cik})
                    MATCH (a:Article)-[:MENTIONS_ORGANIZATION]->(o)
                    OPTIONAL MATCH (a)-[:HAS_SENTIMENT]->(s:Sentiment)
                    RETURN a.title AS title, a.publishedDate AS published,
                           a.url AS url, a.summary AS summary,
                           s.label AS sentiment_label, s.compound AS sentiment_score
                    ORDER BY a.publishedDate DESC
                    LIMIT $k
                """, cik=cik, k=k//2).data()
            except Exception:
                pass  # News might not be available
            
            return {
                "facts": facts,
                "news": news
            }


class BaselineLLMApproach:
    """Direct LLM approach with 10-K data in context window."""
    
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        self.llm = make_llm_from_env()
    
    def close(self):
        self.driver.close()
    
    def get_raw_10k_data(self, cik: str, limit: int = 20) -> str:
        """Get raw 10-K data for direct LLM context (optimized)."""
        with self.driver.session(database=self.cfg.database) as session:
            # Simplified query to avoid memory issues
            query = """
            MATCH (o:Organization {cik: $cik})-[:FILED]->(f:Filing)
            MATCH (f)-[:HAS_FACT]->(fact:Fact)-[:OF_CONCEPT]->(c:Concept)
            MATCH (fact)-[:FOR_PERIOD]->(p:Period)
            WHERE c.name CONTAINS 'Revenue' OR c.name CONTAINS 'NetIncome' OR 
                  c.name CONTAINS 'TotalAssets' OR c.name CONTAINS 'Cash'
            RETURN 
                f.accession as accession,
                f.filedAt as filedAt,
                c.name as concept,
                fact.value as value,
                CASE WHEN p.end IS NOT NULL THEN p.end 
                     WHEN p.instant IS NOT NULL THEN p.instant 
                     ELSE p.start END AS period
            ORDER BY f.filedAt DESC
            LIMIT $limit
            """
            
            results = session.run(query, cik=cik, limit=limit).data()
            
            # Format as structured text
            context_parts = []
            context_parts.append("=== 10-K FINANCIAL DATA ===")
            
            for result in results:
                context_parts.append(f"\nFiling: {result['accession']} ({result['filedAt']})")
                context_parts.append(f"Concept: {result['concept']}")
                context_parts.append(f"Value: {result['value']}")
                context_parts.append(f"Period: {result['period']}")
                context_parts.append("")
            
            return "\n".join(context_parts)
    
    def query(self, prompt: str, cik: str) -> Tuple[str, str]:
        """Query using direct LLM approach."""
        start_time = time.time()
        
        # Get raw 10-K data (reduced limit)
        raw_data = self.get_raw_10k_data(cik, limit=20)
        
        # Build context
        context = f"""
        <<PROMPT>>
        {prompt}
        
        <<10-K FINANCIAL DATA>>
        {raw_data}
        
        <<INSTRUCTIONS>>
        Answer the question using the provided 10-K financial data. 
        Be specific and cite accession numbers and time periods when making claims.
        """
        
        # Generate response
        system = ("You are a financial analyst. Use the provided 10-K data to answer questions. "
                 "Be precise with numbers and cite specific filings and time periods.")
        
        response = self.llm.generate(system, context)
        
        response_time = time.time() - start_time
        
        return response, context


class GraphRAGApproach:
    """Optimized GraphRAG approach with memory-efficient queries."""
    
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.retriever = OptimizedNeo4jRetriever(cfg)
        self.llm = make_llm_from_env()
    
    def close(self):
        self.retriever.close()
    
    def query(self, prompt: str, cik: str) -> Tuple[str, str, Dict[str, Any]]:
        """Query using optimized GraphRAG approach."""
        start_time = time.time()
        
        # Get optimized context
        context_data = self.retriever.get_optimized_context(prompt, cik, k=20)
        
        # Build enhanced context
        context = self._build_context(prompt, cik, context_data)
        
        # Generate response
        system = ("You are an expert financial analyst and strategic planning assistant. "
                 "You have access to financial data and news sentiment. "
                 "Provide strategic insights that connect quantitative metrics with qualitative context.")
        
        response = self.llm.generate(system, context)
        
        response_time = time.time() - start_time
        
        return response, context, context_data
    
    def _build_context(self, prompt: str, cik: str, context_data: Dict[str, Any]) -> str:
        """Build context from optimized data."""
        context_parts = [f"Query: {prompt}\n"]
        
        # Add facts
        if context_data.get("facts"):
            context_parts.append("=== FINANCIAL FACTS ===")
            for fact in context_data["facts"][:10]:  # Limit to 10 facts
                context_parts.append(f"• {fact['concept']}: {fact['value']} ({fact['period']})")
                context_parts.append(f"  Filing: {fact['accession']} ({fact['filed']})")
            context_parts.append("")
        
        # Add news
        if context_data.get("news"):
            context_parts.append("=== NEWS SENTIMENT ===")
            for article in context_data["news"][:5]:  # Limit to 5 articles
                context_parts.append(f"• {article['title']}")
                if article.get('sentiment_label'):
                    context_parts.append(f"  Sentiment: {article['sentiment_label']} ({article.get('sentiment_score', 'N/A')})")
                context_parts.append(f"  Date: {article['published']}")
            context_parts.append("")
        
        return "\n".join(context_parts)


class OptimizedEvaluationFramework:
    """Memory-optimized evaluation framework."""
    
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.baseline = BaselineLLMApproach(cfg)
        self.graphrag = GraphRAGApproach(cfg)
        self.test_queries = self._load_test_queries()
    
    def close(self):
        self.baseline.close()
        self.graphrag.close()
    
    def _load_test_queries(self) -> List[TestQuery]:
        """Load simplified test queries."""
        return [
            TestQuery(
                id="factual_001",
                query="What was IBM's total revenue in 2023?",
                category="factual",
                expected_elements=["revenue", "2023", "dollars"],
                expected_citations=["accession", "filing date"],
                difficulty="easy",
                description="Basic factual financial query"
            ),
            TestQuery(
                id="analytical_001", 
                query="What are IBM's main revenue sources?",
                category="analytical",
                expected_elements=["revenue", "sources", "business", "segments"],
                expected_citations=["multiple sources", "accession numbers"],
                difficulty="medium",
                description="Revenue source analysis"
            ),
            TestQuery(
                id="strategic_001",
                query="What are IBM's main strategic initiatives?",
                category="strategic", 
                expected_elements=["strategy", "initiative", "business", "direction"],
                expected_citations=["narrative sections", "strategic context"],
                difficulty="medium",
                description="Strategic analysis"
            ),
            TestQuery(
                id="temporal_001",
                query="How has IBM's revenue changed recently?",
                category="temporal",
                expected_elements=["revenue", "change", "trend", "growth"],
                expected_citations=["time periods", "comparisons"],
                difficulty="medium", 
                description="Temporal analysis"
            ),
            TestQuery(
                id="cross_domain_001",
                query="What is the market sentiment around IBM?",
                category="cross_domain",
                expected_elements=["sentiment", "market", "news", "reaction"],
                expected_citations=["news articles", "sentiment scores"],
                difficulty="medium",
                description="Cross-domain analysis"
            )
        ]
    
    def evaluate_single_query(self, test_query: TestQuery, cik: str) -> Tuple[EvaluationResult, EvaluationResult]:
        """Evaluate a single query with both approaches."""
        
        try:
            # Get GraphRAG result
            graphrag_response, graphrag_context, graphrag_data = self.graphrag.query(test_query.query, cik)
            graphrag_metrics = self._calculate_metrics(test_query, graphrag_response, graphrag_context, graphrag_data, "graphrag")
            graphrag_result = EvaluationResult(
                query_id=test_query.id,
                approach="graphrag",
                response=graphrag_response,
                context_used=graphrag_context,
                metrics=graphrag_metrics,
                timestamp=datetime.now(),
                raw_data=graphrag_data
            )
        except Exception as e:
            print(f"  GraphRAG error for {test_query.id}: {e}")
            graphrag_result = EvaluationResult(
                query_id=test_query.id,
                approach="graphrag",
                response=f"Error: {e}",
                context_used="",
                metrics=EvaluationMetrics(),
                timestamp=datetime.now(),
                raw_data={}
            )
        
        try:
            # Get baseline result
            baseline_response, baseline_context = self.baseline.query(test_query.query, cik)
            baseline_metrics = self._calculate_metrics(test_query, baseline_response, baseline_context, {}, "direct_llm")
            baseline_result = EvaluationResult(
                query_id=test_query.id,
                approach="direct_llm", 
                response=baseline_response,
                context_used=baseline_context,
                metrics=baseline_metrics,
                timestamp=datetime.now(),
                raw_data={}
            )
        except Exception as e:
            print(f"  Baseline error for {test_query.id}: {e}")
            baseline_result = EvaluationResult(
                query_id=test_query.id,
                approach="direct_llm",
                response=f"Error: {e}",
                context_used="",
                metrics=EvaluationMetrics(),
                timestamp=datetime.now(),
                raw_data={}
            )
        
        return graphrag_result, baseline_result
    
    def _calculate_metrics(self, test_query: TestQuery, response: str, context: str, 
                          context_data: Dict[str, Any], approach: str) -> EvaluationMetrics:
        """Calculate comprehensive metrics for a response."""
        
        # Basic metrics
        response_time = 0.0  # Will be set by caller
        context_size = len(context.split())
        
        # Factual accuracy (check for expected elements)
        factual_accuracy = self._calculate_factual_accuracy(test_query, response)
        
        # Citation accuracy
        citation_accuracy = self._calculate_citation_accuracy(test_query, response)
        
        # Numerical precision
        numerical_precision = self._calculate_numerical_precision(response)
        
        # Context relevance
        context_relevance = self._calculate_context_relevance(test_query, context)
        
        # Context precision and recall
        context_precision, context_recall = self._calculate_context_precision_recall(test_query, context)
        
        # Response completeness
        completeness_score = self._calculate_completeness(test_query, response)
        
        # Response depth
        depth_score = self._calculate_depth(response)
        
        # Coherence
        coherence_score = self._calculate_coherence(response)
        
        # Advanced capabilities (GraphRAG-specific)
        temporal_analysis_score = 0.0
        cross_reference_score = 0.0
        sentiment_integration_score = 0.0
        strategic_insight_score = 0.0
        
        if approach == "graphrag":
            temporal_analysis_score = self._calculate_temporal_analysis(response, context_data)
            cross_reference_score = self._calculate_cross_reference(response, context_data)
            sentiment_integration_score = self._calculate_sentiment_integration(response, context_data)
            strategic_insight_score = self._calculate_strategic_insight(response, context_data)
        
        # Token efficiency
        token_efficiency = self._calculate_token_efficiency(response, context_size)
        
        # Cost estimation (simplified)
        estimated_cost = self._estimate_cost(response, context_size)
        cost_per_insight = estimated_cost / max(1, len([e for e in test_query.expected_elements if e.lower() in response.lower()]))
        
        return EvaluationMetrics(
            factual_accuracy=factual_accuracy,
            citation_accuracy=citation_accuracy,
            numerical_precision=numerical_precision,
            context_relevance=context_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            completeness_score=completeness_score,
            depth_score=depth_score,
            coherence_score=coherence_score,
            temporal_analysis_score=temporal_analysis_score,
            cross_reference_score=cross_reference_score,
            sentiment_integration_score=sentiment_integration_score,
            strategic_insight_score=strategic_insight_score,
            response_time=response_time,
            context_size=context_size,
            token_efficiency=token_efficiency,
            estimated_cost=estimated_cost,
            cost_per_insight=cost_per_insight
        )
    
    def _calculate_factual_accuracy(self, test_query: TestQuery, response: str) -> float:
        """Calculate factual accuracy based on expected elements."""
        response_lower = response.lower()
        found_elements = sum(1 for element in test_query.expected_elements 
                           if element.lower() in response_lower)
        return found_elements / len(test_query.expected_elements)
    
    def _calculate_citation_accuracy(self, test_query: TestQuery, response: str) -> float:
        """Calculate citation accuracy."""
        citation_indicators = ["accession", "filing", "form", "date", "period", "quarter", "year"]
        found_citations = sum(1 for citation in citation_indicators 
                            if citation.lower() in response.lower())
        return min(1.0, found_citations / len(citation_indicators))
    
    def _calculate_numerical_precision(self, response: str) -> float:
        """Calculate numerical precision (presence of specific numbers)."""
        numbers = re.findall(r'\$?[\d,]+\.?\d*[MBK]?', response)
        return min(1.0, len(numbers) / 3)  # Normalize to 3 numbers as full score
    
    def _calculate_context_relevance(self, test_query: TestQuery, context: str) -> float:
        """Calculate how relevant the context is to the query."""
        query_words = set(test_query.query.lower().split())
        context_words = set(context.lower().split())
        intersection = query_words.intersection(context_words)
        return len(intersection) / len(query_words) if query_words else 0.0
    
    def _calculate_context_precision_recall(self, test_query: TestQuery, context: str) -> Tuple[float, float]:
        """Calculate context precision and recall."""
        context_lower = context.lower()
        relevant_terms = sum(1 for element in test_query.expected_elements 
                           if element.lower() in context_lower)
        precision = relevant_terms / max(1, len(test_query.expected_elements))
        recall = relevant_terms / max(1, len(test_query.expected_elements))
        return precision, recall
    
    def _calculate_completeness(self, test_query: TestQuery, response: str) -> float:
        """Calculate response completeness."""
        response_length = len(response.split())
        expected_lengths = {"easy": 50, "medium": 100, "hard": 150}
        expected_length = expected_lengths.get(test_query.difficulty, 100)
        return min(1.0, response_length / expected_length)
    
    def _calculate_depth(self, response: str) -> float:
        """Calculate response depth (analytical depth indicators)."""
        depth_indicators = ["analysis", "trend", "comparison", "implication", "strategy", "risk", "opportunity"]
        found_indicators = sum(1 for indicator in depth_indicators 
                             if indicator.lower() in response.lower())
        return found_indicators / len(depth_indicators)
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate response coherence (simplified)."""
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.0
        
        transition_words = ["however", "furthermore", "additionally", "moreover", "therefore", "consequently"]
        found_transitions = sum(1 for word in transition_words 
                              if word.lower() in response.lower())
        return min(1.0, found_transitions / 2)  # Normalize
    
    def _calculate_temporal_analysis(self, response: str, context_data: Dict[str, Any]) -> float:
        """Calculate temporal analysis capability."""
        temporal_indicators = ["trend", "over time", "historical", "progression", "evolution", "change"]
        found_indicators = sum(1 for indicator in temporal_indicators 
                             if indicator.lower() in response.lower())
        
        temporal_bonus = 0.2 if context_data.get("facts") else 0.0
        
        return min(1.0, (found_indicators / len(temporal_indicators)) + temporal_bonus)
    
    def _calculate_cross_reference(self, response: str, context_data: Dict[str, Any]) -> float:
        """Calculate cross-reference capability."""
        cross_ref_indicators = ["related", "connected", "linked", "associated", "correlation"]
        found_indicators = sum(1 for indicator in cross_ref_indicators 
                             if indicator.lower() in response.lower())
        
        cross_ref_bonus = 0.3 if context_data.get("news") else 0.0
        
        return min(1.0, (found_indicators / len(cross_ref_indicators)) + cross_ref_bonus)
    
    def _calculate_sentiment_integration(self, response: str, context_data: Dict[str, Any]) -> float:
        """Calculate sentiment integration capability."""
        sentiment_indicators = ["sentiment", "positive", "negative", "market reaction", "perception"]
        found_indicators = sum(1 for indicator in sentiment_indicators 
                             if indicator.lower() in response.lower())
        
        sentiment_bonus = 0.4 if context_data.get("news") else 0.0
        
        return min(1.0, (found_indicators / len(sentiment_indicators)) + sentiment_bonus)
    
    def _calculate_strategic_insight(self, response: str, context_data: Dict[str, Any]) -> float:
        """Calculate strategic insight capability."""
        strategic_indicators = ["strategy", "strategic", "initiative", "recommendation", "outlook", "future"]
        found_indicators = sum(1 for indicator in strategic_indicators 
                             if indicator.lower() in response.lower())
        
        strategic_bonus = 0.3 if context_data.get("facts") else 0.0
        
        return min(1.0, (found_indicators / len(strategic_indicators)) + strategic_bonus)
    
    def _calculate_token_efficiency(self, response: str, context_size: int) -> float:
        """Calculate token efficiency (insights per token)."""
        response_tokens = len(response.split())
        if response_tokens == 0:
            return 0.0
        
        insights = len([word for word in response.split() 
                       if word.lower() in ["analysis", "trend", "insight", "finding", "conclusion"]])
        
        return insights / response_tokens
    
    def _estimate_cost(self, response: str, context_size: int) -> float:
        """Estimate cost (simplified model)."""
        total_tokens = len(response.split()) + context_size
        return (total_tokens / 1000) * 0.01
    
    def run_full_evaluation(self, cik: str) -> Dict[str, Any]:
        """Run full evaluation across all test queries."""
        results = {
            "graphrag_results": [],
            "baseline_results": [],
            "comparison_summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        print("Running optimized evaluation...")
        print("=" * 50)
        
        for i, test_query in enumerate(self.test_queries, 1):
            print(f"\nEvaluating query {i}/{len(self.test_queries)}: {test_query.id}")
            print(f"Query: {test_query.query}")
            
            try:
                graphrag_result, baseline_result = self.evaluate_single_query(test_query, cik)
                
                results["graphrag_results"].append(asdict(graphrag_result))
                results["baseline_results"].append(asdict(baseline_result))
                
                # Print quick comparison
                print(f"  GraphRAG - Accuracy: {graphrag_result.metrics.factual_accuracy:.2f}, "
                      f"Depth: {graphrag_result.metrics.depth_score:.2f}, "
                      f"Sentiment: {graphrag_result.metrics.sentiment_integration_score:.2f}")
                print(f"  Baseline - Accuracy: {baseline_result.metrics.factual_accuracy:.2f}, "
                      f"Depth: {baseline_result.metrics.depth_score:.2f}, "
                      f"Sentiment: {baseline_result.metrics.sentiment_integration_score:.2f}")
                
            except Exception as e:
                print(f"  Error evaluating query {test_query.id}: {e}")
                continue
        
        # Calculate summary statistics
        results["comparison_summary"] = self._calculate_summary_statistics(results)
        
        return results
    
    def _calculate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics comparing both approaches."""
        graphrag_metrics = [r["metrics"] for r in results["graphrag_results"]]
        baseline_metrics = [r["metrics"] for r in results["baseline_results"]]
        
        def avg_metric(metrics_list, metric_name):
            values = []
            for m in metrics_list:
                if isinstance(m, dict):
                    values.append(m.get(metric_name, 0.0))
                else:
                    values.append(getattr(m, metric_name, 0.0))
            return statistics.mean(values) if values else 0.0
        
        summary = {
            "graphrag_averages": {
                "factual_accuracy": avg_metric(graphrag_metrics, "factual_accuracy"),
                "citation_accuracy": avg_metric(graphrag_metrics, "citation_accuracy"),
                "completeness_score": avg_metric(graphrag_metrics, "completeness_score"),
                "depth_score": avg_metric(graphrag_metrics, "depth_score"),
                "temporal_analysis_score": avg_metric(graphrag_metrics, "temporal_analysis_score"),
                "cross_reference_score": avg_metric(graphrag_metrics, "cross_reference_score"),
                "sentiment_integration_score": avg_metric(graphrag_metrics, "sentiment_integration_score"),
                "strategic_insight_score": avg_metric(graphrag_metrics, "strategic_insight_score"),
                "response_time": avg_metric(graphrag_metrics, "response_time"),
                "context_size": avg_metric(graphrag_metrics, "context_size"),
                "estimated_cost": avg_metric(graphrag_metrics, "estimated_cost")
            },
            "baseline_averages": {
                "factual_accuracy": avg_metric(baseline_metrics, "factual_accuracy"),
                "citation_accuracy": avg_metric(baseline_metrics, "citation_accuracy"),
                "completeness_score": avg_metric(baseline_metrics, "completeness_score"),
                "depth_score": avg_metric(baseline_metrics, "depth_score"),
                "temporal_analysis_score": avg_metric(baseline_metrics, "temporal_analysis_score"),
                "cross_reference_score": avg_metric(baseline_metrics, "cross_reference_score"),
                "sentiment_integration_score": avg_metric(baseline_metrics, "sentiment_integration_score"),
                "strategic_insight_score": avg_metric(baseline_metrics, "strategic_insight_score"),
                "response_time": avg_metric(baseline_metrics, "response_time"),
                "context_size": avg_metric(baseline_metrics, "context_size"),
                "estimated_cost": avg_metric(baseline_metrics, "estimated_cost")
            }
        }
        
        # Calculate improvements
        improvements = {}
        for metric in summary["graphrag_averages"]:
            baseline_val = summary["baseline_averages"][metric]
            graphrag_val = summary["graphrag_averages"][metric]
            if baseline_val > 0:
                improvement = ((graphrag_val - baseline_val) / baseline_val) * 100
                improvements[metric] = improvement
            else:
                improvements[metric] = 0.0
        
        summary["improvements"] = improvements
        
        return summary


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG vs Direct LLM approaches (Optimized)")
    parser.add_argument("--cik", type=str, help="Company CIK to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results_optimized.json", help="Output file")
    parser.add_argument("--show-details", action="store_true", help="Show detailed results")
    
    args = parser.parse_args()
    
    # Default to IBM if no CIK provided
    cik = args.cik or "51143"  # IBM's CIK
    
    # Setup configuration
    cfg = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        database=os.getenv("NEO4J_DB", "neo4j"),
    )
    
    # Run evaluation
    framework = OptimizedEvaluationFramework(cfg)
    try:
        results = framework.run_full_evaluation(cik)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("OPTIMIZED EVALUATION COMPLETE")
        print(f"{'='*60}")
        
        # Print summary
        summary = results["comparison_summary"]
        print(f"\nSUMMARY STATISTICS:")
        print(f"GraphRAG vs Baseline Improvements:")
        
        for metric, improvement in summary["improvements"].items():
            print(f"  {metric}: {improvement:+.1f}%")
        
        print(f"\nDetailed results saved to: {args.output}")
        
        if args.show_details:
            print(f"\nDETAILED RESULTS:")
            print(json.dumps(results, indent=2, default=str))
        
    finally:
        framework.close()


if __name__ == "__main__":
    main()
