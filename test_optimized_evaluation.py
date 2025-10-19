#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_optimized_evaluation.py

Test script for the optimized evaluation framework.
"""

import os
import sys
from evaluation_framework_optimized import OptimizedEvaluationFramework, Neo4jConfig

def test_optimized_evaluation():
    """Test the optimized evaluation framework."""
    
    # Setup configuration
    cfg = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        database=os.getenv("NEO4J_DB", "neo4j"),
    )
    
    print("Testing Optimized Evaluation Framework")
    print("=" * 50)
    
    # Test with a single query first
    framework = OptimizedEvaluationFramework(cfg)
    try:
        # Test basic connectivity
        print("Testing Neo4j connectivity...")
        test_query = framework.test_queries[0]  # Get first test query
        print(f"Test query: {test_query.query}")
        
        # Run single evaluation
        print("Running single query evaluation...")
        graphrag_result, baseline_result = framework.evaluate_single_query(test_query, "51143")
        
        print(f"GraphRAG Response: {graphrag_result.response[:200]}...")
        print(f"Baseline Response: {baseline_result.response[:200]}...")
        
        print(f"GraphRAG Metrics:")
        print(f"  Factual Accuracy: {graphrag_result.metrics.factual_accuracy:.2f}")
        print(f"  Depth Score: {graphrag_result.metrics.depth_score:.2f}")
        print(f"  Sentiment Integration: {graphrag_result.metrics.sentiment_integration_score:.2f}")
        
        print(f"Baseline Metrics:")
        print(f"  Factual Accuracy: {baseline_result.metrics.factual_accuracy:.2f}")
        print(f"  Depth Score: {baseline_result.metrics.depth_score:.2f}")
        print(f"  Sentiment Integration: {baseline_result.metrics.sentiment_integration_score:.2f}")
        
        print("\n✅ Single query test passed!")
        
        # Test full evaluation
        print("\nRunning full evaluation...")
        results = framework.run_full_evaluation("51143")
        
        print(f"\nFull evaluation completed!")
        print(f"GraphRAG results: {len(results['graphrag_results'])}")
        print(f"Baseline results: {len(results['baseline_results'])}")
        
        # Print summary
        summary = results["comparison_summary"]
        print(f"\nSummary Statistics:")
        for metric, improvement in summary["improvements"].items():
            print(f"  {metric}: {improvement:+.1f}%")
        
        print("\n✅ Full evaluation test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        framework.close()
    
    return True

if __name__ == "__main__":
    success = test_optimized_evaluation()
    sys.exit(0 if success else 1)
