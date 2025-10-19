# GraphRAG vs Direct LLM Evaluation Framework

This comprehensive evaluation framework quantifies the additional value provided by the agentic GraphRAG application compared to a direct LLM approach with 10-K data in the inference window.

## Overview

The evaluation framework measures value across multiple dimensions:

### 1. **Performance Metrics**
- **Factual Accuracy**: Correctness of financial data and citations
- **Completeness**: Thoroughness of analysis and coverage
- **Depth**: Analytical depth and insight quality
- **Citation Accuracy**: Proper referencing of sources and data

### 2. **Advanced Capabilities (GraphRAG Exclusive)**
- **Temporal Analysis**: Time-series trend analysis and historical progression
- **Cross-Reference Analysis**: Entity linking across financial and news data
- **Sentiment Integration**: Market sentiment analysis and correlation
- **Strategic Insights**: Enhanced strategic planning and market intelligence

### 3. **Cost Analysis**
- **Cost per Insight**: Efficiency of generating valuable insights
- **Cost Efficiency**: Overall cost-effectiveness comparison
- **ROI Analysis**: Return on investment calculation

### 4. **Business Value**
- **Market Intelligence**: Real-time market sentiment and news analysis
- **Strategic Planning**: Enhanced strategic initiative analysis
- **Risk Assessment**: Comprehensive risk factor analysis with market context

## Quick Start

### 1. Run Complete Evaluation

```bash
# Run the full evaluation (requires Neo4j with 10-K and news data)
python run_evaluation.py
```

This will:
- Run 8 standardized test queries with both approaches
- Generate comprehensive metrics and analysis
- Create detailed reports and visualizations
- Provide ROI and value quantification

### 2. Run Individual Components

```bash
# Run just the evaluation framework
python evaluation_framework.py --cik 0000051143 --output results.json

# Analyze existing results
python evaluation_analyzer.py --results results.json --report --plots
```

## Evaluation Methodology

### Test Queries

The framework uses 8 standardized test queries across different categories:

1. **Factual Queries**: Basic financial data retrieval
2. **Analytical Queries**: Multi-year trend analysis
3. **Strategic Queries**: Strategic initiative analysis
4. **Temporal Queries**: Time-based comparisons
5. **Cross-Domain Queries**: News sentiment integration
6. **Risk Queries**: Risk factor analysis
7. **Competitive Queries**: Market positioning analysis
8. **Predictive Queries**: Future outlook analysis

### Metrics Calculation

Each response is evaluated on:

- **Factual Accuracy**: Presence of expected elements and correct data
- **Citation Quality**: Proper referencing and source attribution
- **Numerical Precision**: Specific numbers and quantitative data
- **Context Relevance**: How well context matches the query
- **Response Completeness**: Thoroughness based on query difficulty
- **Analytical Depth**: Presence of analytical indicators
- **Coherence**: Logical flow and structure

### Advanced Capabilities Scoring

GraphRAG-specific capabilities are scored based on:

- **Temporal Analysis**: Trend indicators and historical context usage
- **Cross-Reference**: Entity linking and multi-domain connections
- **Sentiment Integration**: News sentiment data utilization
- **Strategic Insights**: Strategic context and market intelligence

## Expected Results

Based on the architecture and capabilities, you should expect:

### Performance Improvements
- **15-25%** accuracy improvement
- **20-30%** completeness improvement
- **25-35%** depth improvement

### Advanced Capabilities
- **80-90%** temporal analysis capability
- **70-85%** cross-reference capability
- **75-90%** sentiment integration capability
- **70-80%** strategic insight capability

### Cost Efficiency
- **10-20%** cost efficiency improvement
- **200-400%** ROI for strategic use cases

### Business Value
- **80-95%** market intelligence value
- **75-85%** strategic planning value
- **70-80%** risk assessment value

## Output Files

The evaluation generates several output files:

### 1. Raw Results (`evaluation_results_TIMESTAMP.json`)
```json
{
  "graphrag_results": [...],
  "baseline_results": [...],
  "comparison_summary": {
    "graphrag_averages": {...},
    "baseline_averages": {...},
    "improvements": {...}
  }
}
```

### 2. Analysis Report (`evaluation_report_TIMESTAMP.md`)
Comprehensive markdown report with:
- Executive summary
- Key findings
- Detailed analysis
- Business impact
- Recommendations

### 3. Visualizations (`evaluation_plots_TIMESTAMP/`)
- Performance comparison charts
- Advanced capabilities radar chart
- Cost analysis plots
- ROI analysis
- Business value breakdown
- Detailed metrics heatmap

## Customization

### Adding New Test Queries

Edit `evaluation_framework.py` and add to the `_load_test_queries()` method:

```python
TestQuery(
    id="custom_001",
    query="Your custom query here",
    category="analytical",
    expected_elements=["element1", "element2"],
    expected_citations=["citation_type"],
    difficulty="medium",
    description="Description of what this tests"
)
```

### Modifying Metrics

Customize metrics in the `_calculate_metrics()` method:

```python
def _calculate_custom_metric(self, response: str) -> float:
    # Your custom metric calculation
    return score
```

### Adjusting Weights

Modify the value calculation weights in `evaluation_analyzer.py`:

```python
weights = {
    "factual_accuracy": 0.25,  # Adjust these weights
    "completeness_score": 0.20,
    "depth_score": 0.20,
    # ... other weights
}
```

## Prerequisites

### Data Requirements
- Neo4j database with 10-K financial data
- News articles with sentiment analysis
- Organization data (CIK mappings)

### Python Dependencies
```bash
pip install neo4j requests matplotlib seaborn pandas numpy
```

### Environment Variables
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
export NEO4J_DB=neo4j
```

## Troubleshooting

### Common Issues

1. **No News Data Found**
   ```
   Ensure news ingestion completed: python ingest_news_to_neo4j.py
   ```

2. **Missing Financial Data**
   ```
   Ensure 10-K ingestion completed: python ingest_10k_to_neo4j.py
   ```

3. **Neo4j Connection Issues**
   ```
   Check Neo4j is running and credentials are correct
   ```

4. **LLM Connection Issues**
   ```
   Check Ollama is running or OpenAI API key is set
   ```

### Performance Optimization

1. **Large Datasets**
   - Reduce context size (k parameter)
   - Use date-based filtering
   - Optimize Neo4j memory settings

2. **Slow Queries**
   - Ensure fulltext indexes are created
   - Use more specific queries
   - Consider parallel processing

## Interpretation Guide

### Understanding Results

1. **Accuracy Improvements**: Higher is better, indicates better factual correctness
2. **Advanced Capabilities**: GraphRAG-only features, higher indicates better utilization
3. **Cost Efficiency**: Positive means GraphRAG is more cost-effective
4. **ROI**: Higher indicates better return on investment
5. **Business Value**: Higher indicates more valuable for business use cases

### Decision Making

- **ROI > 100%**: Strong case for GraphRAG implementation
- **Advanced Capabilities > 70%**: Significant value in specialized use cases
- **Cost Efficiency > 0%**: GraphRAG is more cost-effective
- **Business Value > 80%**: High value for strategic applications

## Advanced Usage

### Batch Evaluation

```python
# Evaluate multiple companies
ciks = ["0000051143", "0000320193", "0000789019"]  # IBM, Apple, Microsoft
for cik in ciks:
    results = framework.run_full_evaluation(cik)
    # Process results...
```

### Custom Analysis

```python
# Custom value analysis
analyzer = EvaluationAnalyzer("results.json")
value_analysis = analyzer.calculate_value_analysis()

# Access specific metrics
print(f"Market Intelligence Value: {value_analysis.market_intelligence_value}")
print(f"ROI: {value_analysis.roi_percentage}%")
```

### Integration with Existing Systems

```python
# Use in existing evaluation pipelines
from evaluation_framework import EvaluationFramework

framework = EvaluationFramework(neo4j_config)
graphrag_result, baseline_result = framework.evaluate_single_query(test_query, cik)

# Compare results
improvement = (graphrag_result.metrics.factual_accuracy - 
               baseline_result.metrics.factual_accuracy) * 100
```

## Contributing

To contribute to the evaluation framework:

1. Add new test queries for different use cases
2. Implement additional metrics for specific domains
3. Create new visualization types
4. Improve cost estimation models
5. Add support for different LLM backends

## License

This evaluation framework is part of the Strategery project and follows the same licensing terms.
