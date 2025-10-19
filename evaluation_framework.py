#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_framework.py

Comprehensive evaluation framework to quantify the additional value of the agentic
GraphRAG application versus a direct LLM approach with 10-K data in the inference window.

This framework measures:
1. Accuracy and Factual Correctness
2. Context Relevance and Precision
3. Response Completeness and Depth
4. Temporal and Cross-Reference Analysis
5. Cost and Performance Metrics
6. Market Intelligence Integration
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


class BaselineLLMApproach:
    """Direct LLM approach with 10-K data in context window."""
    
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        self.llm = make_llm_from_env()
    
    def close(self):
        self.driver.close()
    
    def get_raw_10k_data(self, cik: str, limit: int = 100) -> str:
        """Get raw 10-K data for direct LLM context."""
        with self.driver.session(database=self.cfg.database) as session:
            # Get comprehensive 10-K data
            query = """
            MATCH (o:Organization {cik: $cik})-[:FILED]->(f:Filing)
            MATCH (f)-[:HAS_FACT]->(fact:Fact)-[:OF_CONCEPT]->(c:Concept)
            MATCH (fact)-[:FOR_PERIOD]->(p:Period)
            OPTIONAL MATCH (sec:Section)-[:HAS_SENTENCE]->(s:Sentence)
            WHERE sec.filingAccession = f.accession
            RETURN 
                f.accession as accession,
                f.filedAt as filedAt,
                f.formType as formType,
                c.name as concept,
                fact.value as value,
                p.start as period_start,
                p.end as period_end,
                p.instant as period_instant,
                collect(DISTINCT s.text)[..5] as sample_sentences
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
                context_parts.append(f"Period: {result['period_start']} to {result['period_end'] or result['period_instant']}")
                
                if result['sample_sentences']:
                    context_parts.append("Sample narrative:")
                    for sentence in result['sample_sentences']:
                        if sentence:
                            context_parts.append(f"  - {sentence[:200]}...")
                context_parts.append("")
            
            return "\n".join(context_parts)
    
    def query(self, prompt: str, cik: str) -> Tuple[str, str]:
        """Query using direct LLM approach."""
        start_time = time.time()
        
        # Get raw 10-K data
        raw_data = self.get_raw_10k_data(cik, limit=50)
        
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
    """Enhanced GraphRAG approach with news integration."""
    
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.retriever = Neo4jRetriever(cfg)
        self.llm = make_llm_from_env()
    
    def close(self):
        self.retriever.close()
    
    def query(self, prompt: str, cik: str) -> Tuple[str, str, Dict[str, Any]]:
        """Query using enhanced GraphRAG approach."""
        start_time = time.time()
        
        # Get comprehensive context
        context_data = self.retriever.get_comprehensive_context(prompt, cik=cik, k=50)
        
        # Build enhanced context
        context = build_enhanced_context(prompt, cik, None, context_data)
        
        # Generate response
        system = ("You are an expert financial analyst and strategic planning assistant. "
                 "You have access to comprehensive financial data, narrative disclosures, temporal trends, "
                 "news articles with sentiment analysis, and cross-referenced market intelligence. "
                 "Provide strategic insights that connect quantitative metrics with qualitative context and market sentiment.")
        
        response = self.llm.generate(system, context)
        
        response_time = time.time() - start_time
        
        return response, context, context_data


class EvaluationFramework:
    """Main evaluation framework."""
    
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.baseline = BaselineLLMApproach(cfg)
        self.graphrag = GraphRAGApproach(cfg)
        self.test_queries = self._load_test_queries()
    
    def close(self):
        self.baseline.close()
        self.graphrag.close()
    
    def _load_test_queries(self) -> List[TestQuery]:
        """Load standardized test queries."""
        return [
            TestQuery(
                id="factual_001",
                query="What was IBM's total revenue in 2023?",
                category="factual",
                expected_elements=["revenue", "2023", "dollars", "million"],
                expected_citations=["accession", "filing date"],
                difficulty="easy",
                description="Basic factual financial query"
            ),
            TestQuery(
                id="analytical_001", 
                query="Analyze IBM's revenue trends over the past 3 years",
                category="analytical",
                expected_elements=["trend", "growth", "decline", "percentage", "comparison"],
                expected_citations=["multiple years", "accession numbers"],
                difficulty="medium",
                description="Multi-year trend analysis"
            ),
            TestQuery(
                id="strategic_001",
                query="What are IBM's main strategic initiatives and how are they performing?",
                category="strategic", 
                expected_elements=["strategy", "initiative", "performance", "investment"],
                expected_citations=["narrative sections", "strategic context"],
                difficulty="hard",
                description="Strategic analysis requiring narrative understanding"
            ),
            TestQuery(
                id="temporal_001",
                query="How did IBM's financial performance change from Q1 to Q4 in 2023?",
                category="temporal",
                expected_elements=["quarterly", "Q1", "Q4", "change", "comparison"],
                expected_citations=["quarterly data", "period comparisons"],
                difficulty="medium", 
                description="Quarterly temporal analysis"
            ),
            TestQuery(
                id="cross_domain_001",
                query="How did market sentiment around IBM change during their recent earnings announcements?",
                category="cross_domain",
                expected_elements=["sentiment", "earnings", "market", "news", "reaction"],
                expected_citations=["news articles", "sentiment scores", "filing dates"],
                difficulty="hard",
                description="Cross-domain analysis requiring news integration"
            ),
            TestQuery(
                id="risk_001",
                query="What are IBM's main risk factors and how have they evolved?",
                category="analytical",
                expected_elements=["risk", "factor", "challenge", "uncertainty"],
                expected_citations=["risk factors", "narrative sections"],
                difficulty="medium",
                description="Risk factor analysis"
            ),
            TestQuery(
                id="competitive_001", 
                query="How does IBM's cloud business performance compare to industry trends?",
                category="strategic",
                expected_elements=["cloud", "business", "performance", "industry", "comparison"],
                expected_citations=["business segments", "market context"],
                difficulty="hard",
                description="Competitive analysis requiring market context"
            ),
            TestQuery(
                id="predictive_001",
                query="Based on current trends, what are IBM's growth prospects?",
                category="strategic",
                expected_elements=["growth", "prospects", "trends", "future", "outlook"],
                expected_citations=["historical trends", "strategic initiatives"],
                difficulty="hard",
                description="Predictive analysis requiring trend synthesis"
            )
        ]
    
    def evaluate_single_query(self, test_query: TestQuery, cik: str) -> Tuple[EvaluationResult, EvaluationResult]:
        """Evaluate a single query with both approaches."""
        
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
        return min(1.0, len(numbers) / 5)  # Normalize to 5 numbers as full score
    
    def _calculate_context_relevance(self, test_query: TestQuery, context: str) -> float:
        """Calculate how relevant the context is to the query."""
        query_words = set(test_query.query.lower().split())
        context_words = set(context.lower().split())
        intersection = query_words.intersection(context_words)
        return len(intersection) / len(query_words) if query_words else 0.0
    
    def _calculate_context_precision_recall(self, test_query: TestQuery, context: str) -> Tuple[float, float]:
        """Calculate context precision and recall."""
        # Simplified implementation
        context_lower = context.lower()
        relevant_terms = sum(1 for element in test_query.expected_elements 
                           if element.lower() in context_lower)
        precision = relevant_terms / max(1, len(test_query.expected_elements))
        recall = relevant_terms / max(1, len(test_query.expected_elements))
        return precision, recall
    
    def _calculate_completeness(self, test_query: TestQuery, response: str) -> float:
        """Calculate response completeness."""
        response_length = len(response.split())
        # Expected length based on difficulty
        expected_lengths = {"easy": 100, "medium": 200, "hard": 300}
        expected_length = expected_lengths.get(test_query.difficulty, 200)
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
        
        # Check for transition words and logical flow
        transition_words = ["however", "furthermore", "additionally", "moreover", "therefore", "consequently"]
        found_transitions = sum(1 for word in transition_words 
                              if word.lower() in response.lower())
        return min(1.0, found_transitions / 3)  # Normalize
    
    def _calculate_temporal_analysis(self, response: str, context_data: Dict[str, Any]) -> float:
        """Calculate temporal analysis capability."""
        temporal_indicators = ["trend", "over time", "historical", "progression", "evolution", "change"]
        found_indicators = sum(1 for indicator in temporal_indicators 
                             if indicator.lower() in response.lower())
        
        # Bonus for actual temporal data usage
        temporal_bonus = 0.2 if context_data.get("facts_with_trends") else 0.0
        
        return min(1.0, (found_indicators / len(temporal_indicators)) + temporal_bonus)
    
    def _calculate_cross_reference(self, response: str, context_data: Dict[str, Any]) -> float:
        """Calculate cross-reference capability."""
        cross_ref_indicators = ["related", "connected", "linked", "associated", "correlation"]
        found_indicators = sum(1 for indicator in cross_ref_indicators 
                             if indicator.lower() in response.lower())
        
        # Bonus for actual cross-domain data
        cross_ref_bonus = 0.3 if context_data.get("news_with_sentiment") else 0.0
        
        return min(1.0, (found_indicators / len(cross_ref_indicators)) + cross_ref_bonus)
    
    def _calculate_sentiment_integration(self, response: str, context_data: Dict[str, Any]) -> float:
        """Calculate sentiment integration capability."""
        sentiment_indicators = ["sentiment", "positive", "negative", "market reaction", "perception"]
        found_indicators = sum(1 for indicator in sentiment_indicators 
                             if indicator.lower() in response.lower())
        
        # Bonus for actual sentiment data usage
        sentiment_bonus = 0.4 if context_data.get("news_with_sentiment", {}).get("articles") else 0.0
        
        return min(1.0, (found_indicators / len(sentiment_indicators)) + sentiment_bonus)
    
    def _calculate_strategic_insight(self, response: str, context_data: Dict[str, Any]) -> float:
        """Calculate strategic insight capability."""
        strategic_indicators = ["strategy", "strategic", "initiative", "recommendation", "outlook", "future"]
        found_indicators = sum(1 for indicator in strategic_indicators 
                             if indicator.lower() in response.lower())
        
        # Bonus for strategic context usage
        strategic_bonus = 0.3 if context_data.get("strategic_context") else 0.0
        
        return min(1.0, (found_indicators / len(strategic_indicators)) + strategic_bonus)
    
    def _calculate_token_efficiency(self, response: str, context_size: int) -> float:
        """Calculate token efficiency (insights per token)."""
        response_tokens = len(response.split())
        if response_tokens == 0:
            return 0.0
        
        # Count insights (simplified)
        insights = len([word for word in response.split() 
                       if word.lower() in ["analysis", "trend", "insight", "finding", "conclusion"]])
        
        return insights / response_tokens
    
    def _estimate_cost(self, response: str, context_size: int) -> float:
        """Estimate cost (simplified model)."""
        # Rough estimate: $0.01 per 1K tokens
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
        
        print("Running comprehensive evaluation...")
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
            values = [getattr(m, metric_name) for m in metrics_list if hasattr(m, metric_name)]
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
    
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG vs Direct LLM approaches")
    parser.add_argument("--cik", type=str, help="Company CIK to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    parser.add_argument("--show-details", action="store_true", help="Show detailed results")
    
    args = parser.parse_args()
    
    # Default to IBM if no CIK provided (without leading zeros to match database)
    cik = args.cik or "51143"  # IBM's CIK
    
    # Setup configuration
    cfg = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        database=os.getenv("NEO4J_DB", "neo4j"),
    )
    
    # Run evaluation
    framework = EvaluationFramework(cfg)
    try:
        results = framework.run_full_evaluation(cik)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
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
