#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_evaluation_framework.py

Advanced semantic evaluation framework using proper NLP techniques
to measure depth, analytical quality, and semantic richness.
"""

import json
import os
import time
import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import statistics
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import spacy
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")
    pass

# Import our existing components
from kg_query import Neo4jRetriever, Neo4jConfig, make_llm_from_env, build_enhanced_context
from neo4j import GraphDatabase


@dataclass
class SemanticMetrics:
    """Advanced semantic metrics for response analysis."""
    
    # Semantic Depth Metrics
    semantic_depth_score: float = 0.0
    analytical_complexity: float = 0.0
    conceptual_density: float = 0.0
    argumentation_strength: float = 0.0
    
    # Topic Modeling Metrics
    topic_diversity: float = 0.0
    topic_coherence: float = 0.0
    semantic_richness: float = 0.0
    
    # Sentiment & Emotion Analysis
    sentiment_polarity: float = 0.0
    sentiment_subjectivity: float = 0.0
    emotional_intensity: float = 0.0
    
    # Linguistic Complexity
    lexical_diversity: float = 0.0
    syntactic_complexity: float = 0.0
    readability_score: float = 0.0
    
    # Domain-Specific Analysis
    financial_terminology_density: float = 0.0
    strategic_language_score: float = 0.0
    risk_assessment_indicators: float = 0.0
    
    # Comparative Metrics
    semantic_similarity_to_expected: float = 0.0
    information_novelty: float = 0.0
    contextual_relevance: float = 0.0


class SemanticAnalyzer:
    """Advanced semantic analysis using NLP techniques."""
    
    def __init__(self):
        """Initialize semantic analyzer with NLP models."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Financial and strategic terminology
        self.financial_terms = {
            'revenue', 'profit', 'margin', 'growth', 'investment', 'return', 'risk', 'valuation',
            'earnings', 'cash flow', 'assets', 'liabilities', 'equity', 'debt', 'capital',
            'market share', 'competitive', 'strategy', 'initiative', 'acquisition', 'merger'
        }
        
        self.strategic_terms = {
            'strategy', 'strategic', 'initiative', 'objective', 'goal', 'vision', 'mission',
            'competitive advantage', 'market position', 'stakeholder', 'value proposition',
            'transformation', 'innovation', 'digital', 'technology', 'partnership', 'alliance'
        }
        
        self.risk_terms = {
            'risk', 'uncertainty', 'volatility', 'exposure', 'vulnerability', 'threat',
            'challenge', 'concern', 'mitigation', 'hedge', 'diversification', 'safeguard'
        }
    
    def analyze_semantic_depth(self, response: str) -> float:
        """Calculate semantic depth using multiple NLP techniques."""
        if not response.strip():
            return 0.0
        
        # 1. Sentence complexity analysis
        try:
            sentences = sent_tokenize(response)
        except LookupError:
            # Fallback to simple sentence splitting if NLTK data not available
            sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        complexity_scores = []
        for sentence in sentences:
            # Count clauses, subordinating conjunctions, complex structures
            complexity = self._calculate_sentence_complexity(sentence)
            complexity_scores.append(complexity)
        
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0.0
        
        # 2. Conceptual density (unique concepts per sentence)
        concepts = self._extract_concepts(response)
        concept_density = len(concepts) / len(sentences) if sentences else 0.0
        
        # 3. Argumentation structure
        argumentation_score = self._analyze_argumentation_structure(response)
        
        # 4. Domain-specific depth
        domain_depth = self._analyze_domain_depth(response)
        
        # Combine metrics
        semantic_depth = (
            avg_complexity * 0.3 +
            concept_density * 0.3 +
            argumentation_score * 0.2 +
            domain_depth * 0.2
        )
        
        return min(1.0, semantic_depth)
    
    def _calculate_sentence_complexity(self, sentence: str) -> float:
        """Calculate syntactic complexity of a sentence."""
        if not sentence.strip():
            return 0.0
        
        # Count various complexity indicators
        complexity_indicators = 0
        
        # Subordinating conjunctions
        subordinating = ['because', 'although', 'while', 'since', 'if', 'when', 'where', 'why', 'how']
        complexity_indicators += sum(1 for word in subordinating if word in sentence.lower())
        
        # Coordinating conjunctions
        coordinating = ['and', 'but', 'or', 'nor', 'for', 'yet', 'so']
        complexity_indicators += sum(1 for word in coordinating if word in sentence.lower())
        
        # Relative clauses
        relative_pronouns = ['which', 'that', 'who', 'whom', 'whose', 'where', 'when']
        complexity_indicators += sum(1 for word in relative_pronouns if word in sentence.lower())
        
        # Parenthetical expressions
        complexity_indicators += sentence.count('(') + sentence.count(')')
        
        # Semicolons and colons
        complexity_indicators += sentence.count(';') + sentence.count(':')
        
        # Normalize by sentence length
        word_count = len(sentence.split())
        return complexity_indicators / max(1, word_count)
    
    def _extract_concepts(self, text: str) -> set:
        """Extract key concepts from text."""
        if self.nlp:
            doc = self.nlp(text)
            concepts = set()
            
            # Extract noun phrases and named entities
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1:  # Multi-word concepts
                    concepts.add(chunk.text.lower())
            
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'MONEY', 'PERCENT', 'DATE']:
                    concepts.add(ent.text.lower())
            
            return concepts
        else:
            # Fallback: extract capitalized words and phrases
            words = text.split()
            concepts = set()
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 3:
                    concepts.add(word.lower())
                    # Check for multi-word concepts
                    if i < len(words) - 1 and words[i + 1][0].isupper():
                        concepts.add(f"{word.lower()} {words[i + 1].lower()}")
            
            return concepts
    
    def _analyze_argumentation_structure(self, text: str) -> float:
        """Analyze argumentation structure and logical flow."""
        argumentation_indicators = [
            'therefore', 'consequently', 'thus', 'hence', 'accordingly',
            'however', 'nevertheless', 'nonetheless', 'despite', 'although',
            'furthermore', 'moreover', 'additionally', 'in addition',
            'first', 'second', 'third', 'initially', 'subsequently', 'finally',
            'for example', 'for instance', 'specifically', 'particularly',
            'in conclusion', 'to summarize', 'overall', 'in summary'
        ]
        
        found_indicators = sum(1 for indicator in argumentation_indicators 
                             if indicator.lower() in text.lower())
        
        # Normalize by text length
        word_count = len(text.split())
        return found_indicators / max(1, word_count / 50)  # Normalize to ~50 words
    
    def _analyze_domain_depth(self, text: str) -> float:
        """Analyze domain-specific depth (financial, strategic, analytical)."""
        text_lower = text.lower()
        
        # Financial terminology density
        financial_score = sum(1 for term in self.financial_terms 
                           if term in text_lower) / len(self.financial_terms)
        
        # Strategic terminology density
        strategic_score = sum(1 for term in self.strategic_terms 
                            if term in text_lower) / len(self.strategic_terms)
        
        # Risk assessment indicators
        risk_score = sum(1 for term in self.risk_terms 
                       if term in text_lower) / len(self.risk_terms)
        
        # Analytical language indicators
        analytical_indicators = [
            'analysis', 'analyze', 'evaluate', 'assess', 'compare', 'contrast',
            'trend', 'pattern', 'correlation', 'relationship', 'impact', 'effect',
            'implication', 'significance', 'insight', 'finding', 'conclusion'
        ]
        analytical_score = sum(1 for indicator in analytical_indicators 
                             if indicator in text_lower) / len(analytical_indicators)
        
        # Weighted combination
        domain_depth = (
            financial_score * 0.3 +
            strategic_score * 0.3 +
            risk_score * 0.2 +
            analytical_score * 0.2
        )
        
        return min(1.0, domain_depth)
    
    def analyze_topic_diversity(self, response: str) -> float:
        """Analyze topic diversity using LDA topic modeling."""
        if not response.strip():
            return 0.0
        
        # Prepare text for topic modeling
        try:
            sentences = sent_tokenize(response)
        except LookupError:
            sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        # Clean and preprocess
        cleaned_sentences = []
        for sentence in sentences:
            # Remove punctuation and convert to lowercase
            cleaned = re.sub(r'[^\w\s]', '', sentence.lower())
            # Remove stop words
            words = [word for word in cleaned.split() 
                    if word not in self.stop_words and len(word) > 2]
            if words:
                cleaned_sentences.append(' '.join(words))
        
        if len(cleaned_sentences) < 2:
            return 0.0
        
        try:
            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
            
            # Apply LDA
            n_topics = min(3, len(cleaned_sentences))  # Max 3 topics
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(tfidf_matrix)
            
            # Calculate topic diversity
            topic_probs = lda.transform(tfidf_matrix)
            topic_entropy = -np.sum(topic_probs * np.log(topic_probs + 1e-10), axis=1)
            avg_entropy = np.mean(topic_entropy)
            
            return min(1.0, avg_entropy / np.log(n_topics))  # Normalize by max entropy
        
        except Exception:
            # Fallback: simple word diversity
            all_words = ' '.join(cleaned_sentences).split()
            unique_words = set(all_words)
            return len(unique_words) / max(1, len(all_words))
    
    def analyze_sentiment_depth(self, response: str) -> Tuple[float, float, float]:
        """Analyze sentiment polarity, subjectivity, and emotional intensity."""
        if not response.strip():
            return 0.0, 0.0, 0.0
        
        # TextBlob sentiment
        blob = TextBlob(response)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # VADER sentiment for more nuanced analysis
        vader_scores = self.sentiment_analyzer.polarity_scores(response)
        compound_score = vader_scores['compound']
        
        # Emotional intensity (absolute value of sentiment)
        emotional_intensity = abs(compound_score)
        
        return polarity, subjectivity, emotional_intensity
    
    def analyze_linguistic_complexity(self, response: str) -> Tuple[float, float, float]:
        """Analyze lexical diversity, syntactic complexity, and readability."""
        if not response.strip():
            return 0.0, 0.0, 0.0
        
        try:
            words = word_tokenize(response.lower())
            sentences = sent_tokenize(response)
        except LookupError:
            words = response.lower().split()
            sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        # Lexical diversity (Type-Token Ratio)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / max(1, len(words))
        
        # Syntactic complexity (average sentence length, clause density)
        avg_sentence_length = len(words) / max(1, len(sentences))
        
        # Simple readability approximation (Flesch-like)
        avg_syllables = self._estimate_syllables(response)
        readability = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables / len(words)
        readability = max(0, min(100, readability)) / 100  # Normalize to 0-1
        
        return lexical_diversity, avg_sentence_length / 20, readability  # Normalize sentence length
    
    def _estimate_syllables(self, text: str) -> int:
        """Estimate syllable count for readability calculation."""
        words = re.findall(r'\b\w+\b', text.lower())
        syllable_count = 0
        
        for word in words:
            # Simple syllable estimation
            vowels = 'aeiouy'
            syllable_count += 1  # At least one syllable per word
            
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels:
                    syllable_count += 1
            
            # Subtract silent 'e'
            if word.endswith('e'):
                syllable_count -= 1
        
        return max(1, syllable_count)
    
    def calculate_semantic_similarity(self, response: str, expected_elements: List[str]) -> float:
        """Calculate semantic similarity to expected elements."""
        if not response.strip() or not expected_elements:
            return 0.0
        
        # Combine expected elements into a single text
        expected_text = ' '.join(expected_elements)
        
        # Vectorize both texts
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([response, expected_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def analyze_comprehensive_semantics(self, response: str, expected_elements: List[str] = None) -> SemanticMetrics:
        """Perform comprehensive semantic analysis."""
        
        # Semantic depth
        semantic_depth = self.analyze_semantic_depth(response)
        
        # Topic diversity
        topic_diversity = self.analyze_topic_diversity(response)
        
        # Sentiment analysis
        polarity, subjectivity, emotional_intensity = self.analyze_sentiment_depth(response)
        
        # Linguistic complexity
        lexical_diversity, syntactic_complexity, readability = self.analyze_linguistic_complexity(response)
        
        # Domain-specific analysis
        financial_density = self._analyze_domain_depth(response)
        strategic_score = self._analyze_strategic_language(response)
        risk_indicators = self._analyze_risk_assessment(response)
        
        # Comparative metrics
        semantic_similarity = self.calculate_semantic_similarity(response, expected_elements or [])
        information_novelty = self._calculate_information_novelty(response)
        contextual_relevance = self._calculate_contextual_relevance(response)
        
        return SemanticMetrics(
            semantic_depth_score=semantic_depth,
            analytical_complexity=semantic_depth,  # Same as semantic depth for now
            conceptual_density=self._calculate_conceptual_density(response),
            argumentation_strength=self._analyze_argumentation_structure(response),
            topic_diversity=topic_diversity,
            topic_coherence=topic_diversity,  # Simplified
            semantic_richness=semantic_depth * topic_diversity,
            sentiment_polarity=polarity,
            sentiment_subjectivity=subjectivity,
            emotional_intensity=emotional_intensity,
            lexical_diversity=lexical_diversity,
            syntactic_complexity=syntactic_complexity,
            readability_score=readability,
            financial_terminology_density=financial_density,
            strategic_language_score=strategic_score,
            risk_assessment_indicators=risk_indicators,
            semantic_similarity_to_expected=semantic_similarity,
            information_novelty=information_novelty,
            contextual_relevance=contextual_relevance
        )
    
    def _analyze_strategic_language(self, text: str) -> float:
        """Analyze strategic language usage."""
        text_lower = text.lower()
        strategic_count = sum(1 for term in self.strategic_terms if term in text_lower)
        return min(1.0, strategic_count / len(self.strategic_terms))
    
    def _analyze_risk_assessment(self, text: str) -> float:
        """Analyze risk assessment indicators."""
        text_lower = text.lower()
        risk_count = sum(1 for term in self.risk_terms if term in text_lower)
        return min(1.0, risk_count / len(self.risk_terms))
    
    def _calculate_conceptual_density(self, text: str) -> float:
        """Calculate conceptual density (concepts per sentence)."""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        concepts = self._extract_concepts(text)
        return len(concepts) / len(sentences)
    
    def _calculate_information_novelty(self, text: str) -> float:
        """Calculate information novelty (simplified)."""
        # This would ideally compare against a corpus of similar texts
        # For now, use lexical diversity as a proxy
        words = word_tokenize(text.lower())
        unique_words = set(words)
        return len(unique_words) / max(1, len(words))
    
    def _calculate_contextual_relevance(self, text: str) -> float:
        """Calculate contextual relevance (simplified)."""
        # This would ideally compare against the query context
        # For now, use presence of financial/strategic terms
        text_lower = text.lower()
        relevant_terms = self.financial_terms.union(self.strategic_terms)
        found_terms = sum(1 for term in relevant_terms if term in text_lower)
        return min(1.0, found_terms / len(relevant_terms))


class SemanticEvaluationFramework:
    """Enhanced evaluation framework with semantic analysis."""
    
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.semantic_analyzer = SemanticAnalyzer()
        # Import existing framework components
        from evaluation_framework_optimized import OptimizedEvaluationFramework
        self.base_framework = OptimizedEvaluationFramework(cfg)
    
    def close(self):
        self.base_framework.close()
    
    def run_semantic_evaluation(self, cik: str) -> Dict[str, Any]:
        """Run evaluation with semantic analysis."""
        print("Running semantic evaluation...")
        print("=" * 50)
        
        # Get base results
        base_results = self.base_framework.run_full_evaluation(cik)
        
        # Enhance with semantic analysis
        enhanced_results = self._enhance_with_semantic_analysis(base_results)
        
        return enhanced_results
    
    def _enhance_with_semantic_analysis(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance base results with semantic analysis."""
        
        # Analyze GraphRAG results
        for result in base_results["graphrag_results"]:
            response = result["response"]
            expected_elements = self._get_expected_elements_for_query(result["query_id"])
            
            semantic_metrics = self.semantic_analyzer.analyze_comprehensive_semantics(
                response, expected_elements
            )
            
            # Add semantic metrics to existing metrics
            result["semantic_metrics"] = asdict(semantic_metrics)
        
        # Analyze baseline results
        for result in base_results["baseline_results"]:
            response = result["response"]
            expected_elements = self._get_expected_elements_for_query(result["query_id"])
            
            semantic_metrics = self.semantic_analyzer.analyze_comprehensive_semantics(
                response, expected_elements
            )
            
            # Add semantic metrics to existing metrics
            result["semantic_metrics"] = asdict(semantic_metrics)
        
        # Calculate semantic comparison summary
        base_results["semantic_comparison"] = self._calculate_semantic_comparison(base_results)
        
        return base_results
    
    def _get_expected_elements_for_query(self, query_id: str) -> List[str]:
        """Get expected elements for a query."""
        expected_mapping = {
            "factual_001": ["revenue", "2023", "dollars", "billion"],
            "analytical_001": ["revenue", "sources", "business", "segments"],
            "strategic_001": ["strategy", "initiative", "business", "direction"],
            "temporal_001": ["revenue", "change", "trend", "growth"],
            "cross_domain_001": ["sentiment", "market", "news", "reaction"]
        }
        return expected_mapping.get(query_id, [])
    
    def _calculate_semantic_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate semantic comparison between approaches."""
        
        # Extract semantic metrics
        graphrag_semantic = [r["semantic_metrics"] for r in results["graphrag_results"]]
        baseline_semantic = [r["semantic_metrics"] for r in results["baseline_results"]]
        
        # Calculate averages for each semantic metric
        semantic_metrics = [
            'semantic_depth_score', 'analytical_complexity', 'conceptual_density',
            'argumentation_strength', 'topic_diversity', 'semantic_richness',
            'sentiment_polarity', 'sentiment_subjectivity', 'emotional_intensity',
            'lexical_diversity', 'syntactic_complexity', 'readability_score',
            'financial_terminology_density', 'strategic_language_score',
            'risk_assessment_indicators', 'semantic_similarity_to_expected',
            'information_novelty', 'contextual_relevance'
        ]
        
        comparison = {
            "graphrag_averages": {},
            "baseline_averages": {},
            "improvements": {}
        }
        
        for metric in semantic_metrics:
            # Calculate averages
            graphrag_avg = np.mean([m.get(metric, 0.0) for m in graphrag_semantic])
            baseline_avg = np.mean([m.get(metric, 0.0) for m in baseline_semantic])
            
            comparison["graphrag_averages"][metric] = graphrag_avg
            comparison["baseline_averages"][metric] = baseline_avg
            
            # Calculate improvement
            if baseline_avg > 0:
                improvement = ((graphrag_avg - baseline_avg) / baseline_avg) * 100
            else:
                improvement = 0.0 if graphrag_avg == 0 else float('inf')
            
            comparison["improvements"][metric] = improvement
        
        return comparison


def main():
    """Main semantic evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run semantic evaluation")
    parser.add_argument("--cik", type=str, default="51143", help="Company CIK")
    parser.add_argument("--output", type=str, default="semantic_evaluation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Setup configuration
    cfg = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
        database=os.getenv("NEO4J_DB", "neo4j"),
    )
    
    # Run semantic evaluation
    framework = SemanticEvaluationFramework(cfg)
    try:
        results = framework.run_semantic_evaluation(args.cik)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("SEMANTIC EVALUATION COMPLETE")
        print(f"{'='*60}")
        
        # Print semantic comparison
        if "semantic_comparison" in results:
            comparison = results["semantic_comparison"]
            print(f"\nSEMANTIC ANALYSIS RESULTS:")
            print(f"GraphRAG vs Baseline Semantic Improvements:")
            
            for metric, improvement in comparison["improvements"].items():
                if improvement != float('inf'):
                    print(f"  {metric}: {improvement:+.1f}%")
                else:
                    print(f"  {metric}: ∞% (baseline was 0)")
        
        print(f"\nDetailed results saved to: {args.output}")
        
    finally:
        framework.close()


if __name__ == "__main__":
    main()
