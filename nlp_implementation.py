# Simplified NLP Implementation for Real Estate Property Descriptions
# Basic text analysis with optional OpenAI enhancement

import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Optional OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not installed. Using basic NLP only.")

class SimplePropertyNLP:
    """
    Simplified NLP processor for real estate descriptions
    """
    
    def __init__(self, use_openai=False, api_key=None):
        """Initialize with optional OpenAI support"""
        self.use_openai = use_openai and OPENAI_AVAILABLE
        
        if self.use_openai and api_key:
            openai.api_key = api_key
            print("OpenAI integration enabled")
        elif self.use_openai:
            print("Warning: OpenAI requested but no API key provided")
            self.use_openai = False
        
        #Simple keyword lists
        self.luxury_words = ['luxury', 'premium', 'elegant', 'stunning', 'beautiful', 'custom', 'designer']
        self.condition_words = ['new', 'renovated', 'updated', 'modern', 'fresh', 'recent']
        self.amenity_words = ['pool', 'spa', 'garage', 'fireplace', 'patio', 'deck', 'garden']
    
    def analyze_with_openai(self, description):
        """Use OpenAI to analyze property description"""
        if not self.use_openai or pd.isna(description):
            return {'openai_sentiment': 0, 'openai_luxury_score': 0, 'openai_appeal': 0}
        
        try:
            prompt = f"""
            Analyze this real estate property description and return ONLY a JSON with these exact keys:
            - sentiment: number from -1 (negative) to 1 (positive)
            - luxury_score: number from 0 to 10 (how luxurious/high-end)
            - appeal: number from 0 to 10 (overall marketing appeal)
            
            Description: {description[:500]}
            
            JSON:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            result = eval(response.choices[0].message.content.strip())
            return {
                'openai_sentiment': result.get('sentiment', 0),
                'openai_luxury_score': result.get('luxury_score', 0),
                'openai_appeal': result.get('appeal', 0)
            }
        except:
            return {'openai_sentiment': 0, 'openai_luxury_score': 0, 'openai_appeal': 0}
    
    def extract_basic_features(self, description):
        """Extract basic NLP features"""
        if pd.isna(description):
            description = ""
        
        text = str(description).lower()
        
        sentiment = TextBlob(description).sentiment.polarity
        
        luxury_count = sum(1 for word in self.luxury_words if word in text)
        condition_count = sum(1 for word in self.condition_words if word in text)
        amenity_count = sum(1 for word in self.amenity_words if word in text)
        
        word_count = len(text.split())
        char_count = len(text)
        
        features = {
            'sentiment': sentiment,
            'luxury_count': luxury_count,
            'condition_count': condition_count,
            'amenity_count': amenity_count,
            'word_count': word_count,
            'char_count': char_count,
            'exclamation_count': text.count('!')
        }
        
        if self.use_openai:
            openai_features = self.analyze_with_openai(description)
            features.update(openai_features)
        
        return features
    
    def process_descriptions(self, descriptions):
        """Process multiple descriptions and return feature matrix"""
        features_list = []
        
        for i, desc in enumerate(descriptions):
            if i % 100 == 0:
                print(f"Processing description {i+1}/{len(descriptions)}")
            
            features = self.extract_basic_features(desc)
            features_list.append(features)
        
        return pd.DataFrame(features_list)

def analyze_nlp_features(use_openai=False, api_key=None, sample_size=1000):
    """
    Analyze NLP features and their correlation with price ranges
    """
    print("=" * 50)
    print("NLP FEATURE ANALYSIS")
    print("=" * 50)
    
    df = pd.read_csv('data.csv')
    
    if sample_size < len(df):
        sample_df = df.sample(n=sample_size, random_state=42)
        print(f"Analyzing {sample_size} random properties")
    else:
        sample_df = df
        print(f"Analyzing all {len(df)} properties")
    
    print("\nExtracting NLP features...")
    nlp_processor = SimplePropertyNLP(use_openai=use_openai, api_key=api_key)
    nlp_features = nlp_processor.process_descriptions(sample_df['description'])
    
    nlp_features['priceRange'] = sample_df['priceRange'].values
    
    price_mapping = {'0-250000': 1, '250000-350000': 2, '350000-450000': 3, 
                    '450000-650000': 4, '650000+': 5}
    nlp_features['price_numeric'] = nlp_features['priceRange'].map(price_mapping)
    
    print(f"\nNLP Feature Correlations with Price Range:")
    print("-" * 50)
    
    feature_cols = [col for col in nlp_features.columns if col not in ['priceRange', 'price_numeric']]
    correlations = {}
    
    for col in feature_cols:
        if nlp_features[col].var() > 0:
            corr = nlp_features[col].corr(nlp_features['price_numeric'])
            correlations[col] = corr
            print(f"{col:20s}: {corr:6.3f}")
    
    print(f"\nNLP Features by Price Range:")
    print("-" * 50)
    
    summary = nlp_features.groupby('priceRange')[feature_cols].mean()
    
    print(f"\nLuxury Keywords by Price Range:")
    for price_range in sorted(summary.index):
        luxury_score = summary.loc[price_range, 'luxury_count']
        print(f"{price_range:15s}: {luxury_score:.2f}")
    
    print(f"\nSentiment by Price Range:")
    for price_range in sorted(summary.index):
        sentiment = summary.loc[price_range, 'sentiment']
        print(f"{price_range:15s}: {sentiment:.3f}")
    
    if use_openai and 'openai_luxury_score' in feature_cols:
        print(f"\nOpenAI Luxury Score by Price Range:")
        for price_range in sorted(summary.index):
            luxury_score = summary.loc[price_range, 'openai_luxury_score']
            print(f"{price_range:15s}: {luxury_score:.2f}")
    
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nMost Predictive NLP Features:")
    print("-" * 50)
    for feature, corr in sorted_correlations[:5]:
        print(f"{feature:20s}: {corr:6.3f}")
    
    # Save results
    output_file = 'nlp_analysis_results.csv'
    nlp_features.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return nlp_features, correlations

def main():
    """
    Main NLP analysis function
    """
    print("=" * 50)
    print("REAL ESTATE NLP ANALYSIS")
    print("=" * 50)
    
    use_openai = input("Use OpenAI for enhanced analysis? (y/n): ").lower() == 'y'
    api_key = None
    
    if use_openai:
        api_key = input("Enter OpenAI API key: ").strip()
        if not api_key:
            use_openai = False
            print("No API key provided, using basic NLP only")
    
    sample_size = input("Enter sample size (or press Enter for 1000): ").strip()
    try:
        sample_size = int(sample_size) if sample_size else 1000
    except:
        sample_size = 1000
    
    nlp_features, correlations = analyze_nlp_features(
        use_openai=use_openai, 
        api_key=api_key,
        sample_size=sample_size
    )
    
    print(f"\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"NLP features extracted and analyzed")
    print(f"Results saved to nlp_analysis_results.csv")
    print(f"Feature correlations calculated")
    if use_openai:
        print(f"AI analysis included")
    print("=" * 50)
    
    return nlp_features

if __name__ == "__main__":
    features = main() 