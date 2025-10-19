#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_semantic_analysis.py

Test script to verify semantic analysis functionality
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_evaluation_framework import SemanticAnalyzer


def test_semantic_analysis():
    """Test semantic analysis with sample responses."""
    
    print("Testing Semantic Analysis Framework")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SemanticAnalyzer()
    
    # Sample responses for testing
    simple_response = """
    IBM's revenue in 2023 was $30.75 billion. This represents growth from the previous year.
    The company reported strong performance across all segments.
    """
    
    complex_response = """
    **IBM's Strategic Revenue Analysis for FY 2023**
    
    IBM's consolidated revenue of $30.75 billion in 2023 represents a strategic pivot toward 
    high-margin cloud services and AI-enhanced workloads. This 1.8% year-over-year growth, 
    while modest, indicates the company's successful transformation from legacy hardware 
    to cloud-native solutions.
    
    **Strategic Implications:**
    
    The revenue growth trajectory suggests several key strategic insights:
    1. **Market Position**: IBM is holding its own against aggressive cloud competitors like 
       Microsoft, Amazon, and Google through differentiated AI and hybrid cloud offerings.
    2. **Risk Assessment**: The modest growth rate indicates potential vulnerability to 
       market volatility, requiring continued investment in innovation and operational 
       efficiency.
    3. **Opportunity Analysis**: The forecasted 2024 revenue of $32.23 billion suggests 
       significant upside potential from AI-powered offerings and strategic partnerships.
    
    **Financial Risk Factors:**
    - Dependence on large enterprise clients creates concentration risk
    - Legacy service margins may compress as cloud adoption accelerates
    - Acquisition integration costs could impact short-term profitability
    
    **Strategic Recommendations:**
    - Continue investing in high-margin cloud services
    - Optimize operational cost structures in legacy segments
    - Leverage AI capabilities for competitive differentiation
    """
    
    print("\n1. Testing Simple Response:")
    print("-" * 30)
    simple_metrics = analyzer.analyze_comprehensive_semantics(simple_response)
    print(f"Semantic Depth: {simple_metrics.semantic_depth_score:.3f}")
    print(f"Analytical Complexity: {simple_metrics.analytical_complexity:.3f}")
    print(f"Conceptual Density: {simple_metrics.conceptual_density:.3f}")
    print(f"Argumentation Strength: {simple_metrics.argumentation_strength:.3f}")
    print(f"Topic Diversity: {simple_metrics.topic_diversity:.3f}")
    print(f"Semantic Richness: {simple_metrics.semantic_richness:.3f}")
    
    print("\n2. Testing Complex Response:")
    print("-" * 30)
    complex_metrics = analyzer.analyze_comprehensive_semantics(complex_response)
    print(f"Semantic Depth: {complex_metrics.semantic_depth_score:.3f}")
    print(f"Analytical Complexity: {complex_metrics.analytical_complexity:.3f}")
    print(f"Conceptual Density: {complex_metrics.conceptual_density:.3f}")
    print(f"Argumentation Strength: {complex_metrics.argumentation_strength:.3f}")
    print(f"Topic Diversity: {complex_metrics.topic_diversity:.3f}")
    print(f"Semantic Richness: {complex_metrics.semantic_richness:.3f}")
    
    print("\n3. Comparison:")
    print("-" * 30)
    print(f"Depth Improvement: {((complex_metrics.semantic_depth_score - simple_metrics.semantic_depth_score) / max(0.1, simple_metrics.semantic_depth_score) * 100):+.1f}%")
    print(f"Complexity Improvement: {((complex_metrics.analytical_complexity - simple_metrics.analytical_complexity) / max(0.1, simple_metrics.analytical_complexity) * 100):+.1f}%")
    print(f"Richness Improvement: {((complex_metrics.semantic_richness - simple_metrics.semantic_richness) / max(0.1, simple_metrics.semantic_richness) * 100):+.1f}%")
    
    print("\n4. Domain-Specific Analysis:")
    print("-" * 30)
    print(f"Financial Terminology Density: {complex_metrics.financial_terminology_density:.3f}")
    print(f"Strategic Language Score: {complex_metrics.strategic_language_score:.3f}")
    print(f"Risk Assessment Indicators: {complex_metrics.risk_assessment_indicators:.3f}")
    
    print("\n5. Linguistic Analysis:")
    print("-" * 30)
    print(f"Lexical Diversity: {complex_metrics.lexical_diversity:.3f}")
    print(f"Syntactic Complexity: {complex_metrics.syntactic_complexity:.3f}")
    print(f"Readability Score: {complex_metrics.readability_score:.3f}")
    
    print("\n6. Sentiment Analysis:")
    print("-" * 30)
    print(f"Sentiment Polarity: {complex_metrics.sentiment_polarity:.3f}")
    print(f"Sentiment Subjectivity: {complex_metrics.sentiment_subjectivity:.3f}")
    print(f"Emotional Intensity: {complex_metrics.emotional_intensity:.3f}")
    
    print(f"\n{'='*50}")
    print("SEMANTIC ANALYSIS TEST COMPLETE")
    print(f"{'='*50}")
    
    return {
        "simple": simple_metrics,
        "complex": complex_metrics
    }


if __name__ == "__main__":
    test_semantic_analysis()
