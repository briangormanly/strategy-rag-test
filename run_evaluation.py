#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_evaluation.py

Simple script to run the evaluation framework and demonstrate the value
quantification of GraphRAG vs Direct LLM approaches.
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_framework_optimized import OptimizedEvaluationFramework as EvaluationFramework, Neo4jConfig
from evaluation_analyzer import EvaluationAnalyzer


def main():
    """Run the complete evaluation process."""
    
    print("GraphRAG vs Direct LLM Evaluation")
    print("=" * 50)
    
    # Configuration
    cfg = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        database=os.getenv("NEO4J_DB", "neo4j"),
    )
    
    # Default to IBM CIK (without leading zeros to match database)
    cik = "51143"  # IBM
    
    print(f"Evaluating with CIK: {cik}")
    print(f"Neo4j URI: {cfg.uri}")
    print(f"Database: {cfg.database}")
    
    # Run evaluation
    print("\nRunning evaluation framework...")
    framework = EvaluationFramework(cfg)
    
    try:
        # Run full evaluation
        results = framework.run_full_evaluation(cik)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nEvaluation results saved to: {results_file}")
        
        # Analyze results
        print("\nAnalyzing results...")
        analyzer = EvaluationAnalyzer(results_file)
        
        # Generate report
        report = analyzer.generate_comprehensive_report()
        report_file = f"evaluation_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Analysis report saved to: {report_file}")
        
        # Print summary
        value_analysis = analyzer.calculate_value_analysis()
        
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Accuracy Improvement: {value_analysis.accuracy_improvement:+.1f}%")
        print(f"Completeness Improvement: {value_analysis.completeness_improvement:+.1f}%")
        print(f"Depth Improvement: {value_analysis.depth_improvement:+.1f}%")
        print(f"")
        print(f"Advanced Capabilities (GraphRAG Exclusive):")
        print(f"  Temporal Analysis: {value_analysis.temporal_analysis_value:.1f}/100")
        print(f"  Cross-Reference: {value_analysis.cross_reference_value:.1f}/100")
        print(f"  Sentiment Integration: {value_analysis.sentiment_integration_value:.1f}/100")
        print(f"  Strategic Insights: {value_analysis.strategic_insight_value:.1f}/100")
        print(f"")
        print(f"Cost Analysis:")
        print(f"  GraphRAG Cost per Insight: ${value_analysis.cost_per_insight_graphrag:.4f}")
        print(f"  Baseline Cost per Insight: ${value_analysis.cost_per_insight_baseline:.4f}")
        print(f"  Cost Efficiency Improvement: {value_analysis.cost_efficiency_improvement:+.1f}%")
        print(f"")
        print(f"ROI Analysis:")
        print(f"  Additional Value per Query: {value_analysis.additional_value_per_query:.1f}%")
        print(f"  ROI Percentage: {value_analysis.roi_percentage:.1f}%")
        print(f"")
        print(f"Business Value:")
        print(f"  Market Intelligence: {value_analysis.market_intelligence_value:.1f}/100")
        print(f"  Strategic Planning: {value_analysis.strategic_planning_value:.1f}/100")
        print(f"  Risk Assessment: {value_analysis.risk_assessment_value:.1f}/100")
        
        # Generate visualizations if matplotlib is available
        try:
            print(f"\nGenerating visualizations...")
            analyzer.create_visualizations(f"evaluation_plots_{timestamp}")
            print(f"Visualizations saved to: evaluation_plots_{timestamp}/")
        except ImportError:
            print("Matplotlib not available - skipping visualizations")
            print("Install with: pip install matplotlib seaborn")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Files generated:")
        print(f"  - {results_file} (raw results)")
        print(f"  - {report_file} (analysis report)")
        if os.path.exists(f"evaluation_plots_{timestamp}"):
            print(f"  - evaluation_plots_{timestamp}/ (visualizations)")
        
        print(f"\nKey Takeaways:")
        print(f"1. GraphRAG provides {value_analysis.accuracy_improvement:+.1f}% accuracy improvement")
        print(f"2. Unique advanced capabilities worth {value_analysis.additional_value_per_query:.1f}% additional value")
        print(f"3. {value_analysis.roi_percentage:.1f}% ROI with enhanced market intelligence")
        print(f"4. Significant value in strategic planning and risk assessment")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        framework.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
