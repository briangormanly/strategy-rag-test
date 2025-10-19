#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_analyzer.py

Analysis and visualization tools for evaluating the value of agentic GraphRAG
versus direct LLM approaches. Provides detailed metrics, cost analysis, and
value quantification.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import os
from datetime import datetime


@dataclass
class ValueAnalysis:
    """Comprehensive value analysis results."""
    
    # Core Performance Metrics
    accuracy_improvement: float
    completeness_improvement: float
    depth_improvement: float
    
    # Advanced Capabilities (GraphRAG-only)
    temporal_analysis_value: float
    cross_reference_value: float
    sentiment_integration_value: float
    strategic_insight_value: float
    
    # Cost Analysis
    cost_per_insight_graphrag: float
    cost_per_insight_baseline: float
    cost_efficiency_improvement: float
    
    # ROI Analysis
    additional_value_per_query: float
    break_even_cost_threshold: float
    roi_percentage: float
    
    # Qualitative Benefits
    market_intelligence_value: float
    strategic_planning_value: float
    risk_assessment_value: float


class EvaluationAnalyzer:
    """Analyzes evaluation results and quantifies value."""
    
    def __init__(self, results_file: str):
        """Initialize with evaluation results."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.summary = self.results.get("comparison_summary", {})
        self.graphrag_results = self.results.get("graphrag_results", [])
        self.baseline_results = self.results.get("baseline_results", [])
    
    def calculate_value_analysis(self) -> ValueAnalysis:
        """Calculate comprehensive value analysis."""
        
        # Core performance improvements
        accuracy_improvement = self._calculate_improvement("factual_accuracy")
        completeness_improvement = self._calculate_improvement("completeness_score")
        depth_improvement = self._calculate_improvement("depth_score")
        
        # Advanced capabilities (GraphRAG exclusive)
        temporal_analysis_value = self._get_graphrag_average("temporal_analysis_score")
        cross_reference_value = self._get_graphrag_average("cross_reference_score")
        sentiment_integration_value = self._get_graphrag_average("sentiment_integration_score")
        strategic_insight_value = self._get_graphrag_average("strategic_insight_score")
        
        # Cost analysis
        cost_per_insight_graphrag = self._get_graphrag_average("cost_per_insight")
        cost_per_insight_baseline = self._get_baseline_average("cost_per_insight")
        cost_efficiency_improvement = self._calculate_cost_efficiency_improvement()
        
        # ROI analysis
        additional_value_per_query = self._calculate_additional_value()
        break_even_cost_threshold = self._calculate_break_even_threshold()
        roi_percentage = self._calculate_roi_percentage()
        
        # Qualitative benefits
        market_intelligence_value = self._calculate_market_intelligence_value()
        strategic_planning_value = self._calculate_strategic_planning_value()
        risk_assessment_value = self._calculate_risk_assessment_value()
        
        return ValueAnalysis(
            accuracy_improvement=accuracy_improvement,
            completeness_improvement=completeness_improvement,
            depth_improvement=depth_improvement,
            temporal_analysis_value=temporal_analysis_value,
            cross_reference_value=cross_reference_value,
            sentiment_integration_value=sentiment_integration_value,
            strategic_insight_value=strategic_insight_value,
            cost_per_insight_graphrag=cost_per_insight_graphrag,
            cost_per_insight_baseline=cost_per_insight_baseline,
            cost_efficiency_improvement=cost_efficiency_improvement,
            additional_value_per_query=additional_value_per_query,
            break_even_cost_threshold=break_even_cost_threshold,
            roi_percentage=roi_percentage,
            market_intelligence_value=market_intelligence_value,
            strategic_planning_value=strategic_planning_value,
            risk_assessment_value=risk_assessment_value
        )
    
    def _calculate_improvement(self, metric: str) -> float:
        """Calculate percentage improvement for a metric."""
        baseline_val = self._get_baseline_average(metric)
        graphrag_val = self._get_graphrag_average(metric)
        
        if baseline_val > 0:
            return ((graphrag_val - baseline_val) / baseline_val) * 100
        return 0.0
    
    def _get_graphrag_average(self, metric: str) -> float:
        """Get average value for GraphRAG approach."""
        return self.summary.get("graphrag_averages", {}).get(metric, 0.0)
    
    def _get_baseline_average(self, metric: str) -> float:
        """Get average value for baseline approach."""
        return self.summary.get("baseline_averages", {}).get(metric, 0.0)
    
    def _calculate_cost_efficiency_improvement(self) -> float:
        """Calculate cost efficiency improvement."""
        baseline_cost = self._get_baseline_average("estimated_cost")
        graphrag_cost = self._get_graphrag_average("estimated_cost")
        
        if baseline_cost > 0:
            return ((baseline_cost - graphrag_cost) / baseline_cost) * 100
        return 0.0
    
    def _calculate_additional_value(self) -> float:
        """Calculate additional value per query."""
        # Weighted combination of improvements
        weights = {
            "factual_accuracy": 0.25,
            "completeness_score": 0.20,
            "depth_score": 0.20,
            "temporal_analysis_score": 0.15,
            "cross_reference_score": 0.10,
            "sentiment_integration_score": 0.10
        }
        
        total_value = 0.0
        for metric, weight in weights.items():
            if metric in ["temporal_analysis_score", "cross_reference_score", "sentiment_integration_score"]:
                # These are GraphRAG-only capabilities
                value = self._get_graphrag_average(metric) * weight
            else:
                # These are improvements over baseline
                improvement = self._calculate_improvement(metric)
                value = (improvement / 100) * weight
            
            total_value += value
        
        return total_value * 100  # Convert to percentage
    
    def _calculate_break_even_threshold(self) -> float:
        """Calculate break-even cost threshold."""
        additional_value = self._calculate_additional_value()
        baseline_cost = self._get_baseline_average("estimated_cost")
        
        if additional_value > 0:
            return baseline_cost + (additional_value / 100) * baseline_cost
        return baseline_cost
    
    def _calculate_roi_percentage(self) -> float:
        """Calculate ROI percentage."""
        additional_value = self._calculate_additional_value()
        cost_increase = self._get_graphrag_average("estimated_cost") - self._get_baseline_average("estimated_cost")
        
        if cost_increase > 0:
            return (additional_value / cost_increase) * 100
        return float('inf') if additional_value > 0 else 0.0
    
    def _calculate_market_intelligence_value(self) -> float:
        """Calculate market intelligence value."""
        sentiment_value = self._get_graphrag_average("sentiment_integration_score")
        cross_ref_value = self._get_graphrag_average("cross_reference_score")
        
        # Market intelligence is primarily from sentiment and cross-references
        return (sentiment_value * 0.7 + cross_ref_value * 0.3) * 100
    
    def _calculate_strategic_planning_value(self) -> float:
        """Calculate strategic planning value."""
        strategic_value = self._get_graphrag_average("strategic_insight_score")
        temporal_value = self._get_graphrag_average("temporal_analysis_score")
        depth_value = self._get_graphrag_average("depth_score")
        
        # Strategic planning benefits from insights, temporal analysis, and depth
        return (strategic_value * 0.5 + temporal_value * 0.3 + depth_value * 0.2) * 100
    
    def _calculate_risk_assessment_value(self) -> float:
        """Calculate risk assessment value."""
        cross_ref_value = self._get_graphrag_average("cross_reference_score")
        sentiment_value = self._get_graphrag_average("sentiment_integration_score")
        completeness_value = self._get_graphrag_average("completeness_score")
        
        # Risk assessment benefits from comprehensive cross-references and sentiment
        return (cross_ref_value * 0.4 + sentiment_value * 0.4 + completeness_value * 0.2) * 100
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive evaluation report."""
        value_analysis = self.calculate_value_analysis()
        
        report = f"""
# GraphRAG vs Direct LLM Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This evaluation compares the agentic GraphRAG application against a direct LLM approach 
using 10-K data in the inference window. The analysis reveals significant value 
additions across multiple dimensions.

## Key Findings

### Performance Improvements
- **Accuracy Improvement**: {value_analysis.accuracy_improvement:+.1f}%
- **Completeness Improvement**: {value_analysis.completeness_improvement:+.1f}%
- **Depth Improvement**: {value_analysis.depth_improvement:+.1f}%

### Advanced Capabilities (GraphRAG Exclusive)
- **Temporal Analysis**: {value_analysis.temporal_analysis_value:.1f}/100
- **Cross-Reference Analysis**: {value_analysis.cross_reference_value:.1f}/100
- **Sentiment Integration**: {value_analysis.sentiment_integration_value:.1f}/100
- **Strategic Insights**: {value_analysis.strategic_insight_value:.1f}/100

### Cost Analysis
- **GraphRAG Cost per Insight**: ${value_analysis.cost_per_insight_graphrag:.4f}
- **Baseline Cost per Insight**: ${value_analysis.cost_per_insight_baseline:.4f}
- **Cost Efficiency Improvement**: {value_analysis.cost_efficiency_improvement:+.1f}%

### ROI Analysis
- **Additional Value per Query**: {value_analysis.additional_value_per_query:.1f}%
- **Break-Even Cost Threshold**: ${value_analysis.break_even_cost_threshold:.4f}
- **ROI Percentage**: {value_analysis.roi_percentage:.1f}%

### Qualitative Benefits
- **Market Intelligence Value**: {value_analysis.market_intelligence_value:.1f}/100
- **Strategic Planning Value**: {value_analysis.strategic_planning_value:.1f}/100
- **Risk Assessment Value**: {value_analysis.risk_assessment_value:.1f}/100

## Detailed Analysis

### 1. Accuracy and Factual Correctness
The GraphRAG approach shows {value_analysis.accuracy_improvement:+.1f}% improvement in factual accuracy,
primarily due to:
- Better context retrieval and relevance
- Cross-validation across multiple data sources
- Enhanced citation accuracy with specific accession numbers

### 2. Advanced Analytical Capabilities
GraphRAG provides unique capabilities not available in direct LLM approach:

**Temporal Analysis ({value_analysis.temporal_analysis_value:.1f}/100)**:
- Time-series trend analysis
- Historical progression tracking
- Period-over-period comparisons

**Cross-Reference Analysis ({value_analysis.cross_reference_value:.1f}/100)**:
- Entity linking across financial and news data
- Concept correlation analysis
- Multi-domain relationship mapping

**Sentiment Integration ({value_analysis.sentiment_integration_value:.1f}/100)**:
- Market sentiment analysis
- News sentiment correlation with financial events
- Sentiment timeline around filing dates

**Strategic Insights ({value_analysis.strategic_insight_value:.1f}/100)**:
- Strategic initiative analysis
- Market perception of strategies
- Competitive positioning insights

### 3. Cost-Benefit Analysis

**Cost Structure**:
- GraphRAG: ${value_analysis.cost_per_insight_graphrag:.4f} per insight
- Baseline: ${value_analysis.cost_per_insight_baseline:.4f} per insight
- Efficiency: {value_analysis.cost_efficiency_improvement:+.1f}% improvement

**Value Proposition**:
- Additional value per query: {value_analysis.additional_value_per_query:.1f}%
- Break-even threshold: ${value_analysis.break_even_cost_threshold:.4f}
- ROI: {value_analysis.roi_percentage:.1f}%

### 4. Business Impact

**Market Intelligence** ({value_analysis.market_intelligence_value:.1f}/100):
- Real-time market sentiment tracking
- News sentiment correlation with financial performance
- Market reaction analysis around key events

**Strategic Planning** ({value_analysis.strategic_planning_value:.1f}/100):
- Enhanced strategic initiative analysis
- Market validation of strategic directions
- Competitive positioning insights

**Risk Assessment** ({value_analysis.risk_assessment_value:.1f}/100):
- Comprehensive risk factor analysis
- Market perception of risks
- Cross-domain risk correlation

## Recommendations

### 1. Implementation Strategy
- **High-Value Use Cases**: Focus on strategic analysis, market intelligence, and risk assessment
- **Cost Optimization**: Leverage the {value_analysis.cost_efficiency_improvement:+.1f}% cost efficiency improvement
- **ROI Maximization**: Target queries with high strategic value to maximize {value_analysis.roi_percentage:.1f}% ROI

### 2. Value Quantification
- **Quantitative Benefits**: {value_analysis.additional_value_per_query:.1f}% additional value per query
- **Qualitative Benefits**: Enhanced market intelligence, strategic planning, and risk assessment
- **Competitive Advantage**: Unique capabilities in sentiment integration and cross-domain analysis

### 3. Scaling Considerations
- **Break-Even Point**: ${value_analysis.break_even_cost_threshold:.4f} per query
- **Volume Benefits**: Higher query volumes improve overall ROI
- **Specialized Applications**: Focus on high-value analytical use cases

## Conclusion

The agentic GraphRAG application provides significant value over direct LLM approaches:

1. **Performance**: {value_analysis.accuracy_improvement:+.1f}% accuracy improvement with enhanced depth and completeness
2. **Capabilities**: Unique advanced capabilities in temporal analysis, cross-referencing, sentiment integration, and strategic insights
3. **Efficiency**: {value_analysis.cost_efficiency_improvement:+.1f}% cost efficiency improvement
4. **ROI**: {value_analysis.roi_percentage:.1f}% return on investment
5. **Business Value**: Enhanced market intelligence, strategic planning, and risk assessment capabilities

The additional value justifies the implementation complexity, particularly for strategic analysis, market intelligence, and comprehensive risk assessment use cases.
"""
        
        return report
    
    def create_visualizations(self, output_dir: str = "evaluation_plots"):
        """Create comprehensive visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance Comparison
        self._create_performance_comparison_plot(output_dir)
        
        # 2. Advanced Capabilities
        self._create_advanced_capabilities_plot(output_dir)
        
        # 3. Cost Analysis
        self._create_cost_analysis_plot(output_dir)
        
        # 4. ROI Analysis
        self._create_roi_analysis_plot(output_dir)
        
        # 5. Business Value
        self._create_business_value_plot(output_dir)
        
        # 6. Detailed Metrics
        self._create_detailed_metrics_plot(output_dir)
        
        print(f"Visualizations saved to: {output_dir}/")
    
    def _create_performance_comparison_plot(self, output_dir: str):
        """Create performance comparison visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Core metrics comparison
        metrics = ['factual_accuracy', 'citation_accuracy', 'completeness_score', 'depth_score']
        graphrag_values = [self._get_graphrag_average(m) for m in metrics]
        baseline_values = [self._get_baseline_average(m) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, baseline_values, width, label='Direct LLM', alpha=0.8)
        ax1.bar(x + width/2, graphrag_values, width, label='GraphRAG', alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Core Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvements
        improvements = [self._calculate_improvement(m) for m in metrics]
        colors = ['green' if x > 0 else 'red' for x in improvements]
        
        ax2.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('GraphRAG Improvements over Baseline')
        ax2.set_xticks(range(len(improvements)))
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_advanced_capabilities_plot(self, output_dir: str):
        """Create advanced capabilities visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        capabilities = ['Temporal Analysis', 'Cross-Reference', 'Sentiment Integration', 'Strategic Insights']
        values = [
            self._get_graphrag_average("temporal_analysis_score"),
            self._get_graphrag_average("cross_reference_score"),
            self._get_graphrag_average("sentiment_integration_score"),
            self._get_graphrag_average("strategic_insight_score")
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(capabilities), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, label='GraphRAG Capabilities')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(capabilities)
        ax.set_ylim(0, 1)
        ax.set_title('Advanced Capabilities (GraphRAG Exclusive)', size=16, pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/advanced_capabilities.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cost_analysis_plot(self, output_dir: str):
        """Create cost analysis visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost per insight
        approaches = ['Direct LLM', 'GraphRAG']
        costs = [self._get_baseline_average("cost_per_insight"), self._get_graphrag_average("cost_per_insight")]
        
        bars = ax1.bar(approaches, costs, color=['lightcoral', 'lightblue'], alpha=0.8)
        ax1.set_ylabel('Cost per Insight ($)')
        ax1.set_title('Cost per Insight Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${cost:.4f}', ha='center', va='bottom')
        
        # Cost efficiency
        efficiency_improvement = self._calculate_cost_efficiency_improvement()
        ax2.bar(['Cost Efficiency'], [efficiency_improvement], 
                color='green' if efficiency_improvement > 0 else 'red', alpha=0.7)
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Cost Efficiency Improvement')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cost_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_roi_analysis_plot(self, output_dir: str):
        """Create ROI analysis visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        value_analysis = self.calculate_value_analysis()
        
        # ROI components
        components = ['Additional Value', 'Cost Increase', 'Net ROI']
        values = [
            value_analysis.additional_value_per_query,
            (value_analysis.cost_per_insight_graphrag - value_analysis.cost_per_insight_baseline) * 100,
            value_analysis.roi_percentage
        ]
        colors = ['green', 'red', 'blue']
        
        bars = ax.bar(components, values, color=colors, alpha=0.7)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('ROI Analysis')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roi_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_business_value_plot(self, output_dir: str):
        """Create business value visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        value_analysis = self.calculate_value_analysis()
        
        business_areas = ['Market Intelligence', 'Strategic Planning', 'Risk Assessment']
        values = [
            value_analysis.market_intelligence_value,
            value_analysis.strategic_planning_value,
            value_analysis.risk_assessment_value
        ]
        
        bars = ax.bar(business_areas, values, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        ax.set_ylabel('Value Score (0-100)')
        ax.set_title('Business Value by Area')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/business_value.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_metrics_plot(self, output_dir: str):
        """Create detailed metrics heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        metrics = [
            'factual_accuracy', 'citation_accuracy', 'completeness_score', 'depth_score',
            'temporal_analysis_score', 'cross_reference_score', 'sentiment_integration_score', 'strategic_insight_score'
        ]
        
        approaches = ['Direct LLM', 'GraphRAG']
        data = []
        
        for approach in approaches:
            row = []
            for metric in metrics:
                if approach == 'Direct LLM':
                    value = self._get_baseline_average(metric)
                else:
                    value = self._get_graphrag_average(metric)
                row.append(value)
            data.append(row)
        
        # Create heatmap
        sns.heatmap(data, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=approaches,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=0.5,
                   ax=ax)
        
        ax.set_title('Detailed Metrics Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detailed_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--results", type=str, required=True, help="Evaluation results JSON file")
    parser.add_argument("--output-dir", type=str, default="evaluation_analysis", help="Output directory")
    parser.add_argument("--report", action="store_true", help="Generate text report")
    parser.add_argument("--plots", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = EvaluationAnalyzer(args.results)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.report:
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()
        
        with open(f"{args.output_dir}/evaluation_report.md", 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {args.output_dir}/evaluation_report.md")
    
    if args.plots:
        # Generate visualizations
        analyzer.create_visualizations(f"{args.output_dir}/plots")
        print(f"Visualizations saved to: {args.output_dir}/plots/")
    
    # Print summary
    value_analysis = analyzer.calculate_value_analysis()
    print(f"\n{'='*60}")
    print("EVALUATION ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Accuracy Improvement: {value_analysis.accuracy_improvement:+.1f}%")
    print(f"Depth Improvement: {value_analysis.depth_improvement:+.1f}%")
    print(f"ROI: {value_analysis.roi_percentage:.1f}%")
    print(f"Market Intelligence Value: {value_analysis.market_intelligence_value:.1f}/100")
    print(f"Strategic Planning Value: {value_analysis.strategic_planning_value:.1f}/100")
    print(f"Risk Assessment Value: {value_analysis.risk_assessment_value:.1f}/100")


if __name__ == "__main__":
    main()
