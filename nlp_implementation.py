# Simplified NLP Implementation for Real Estate Property Descriptions
# Basic text analysis with optional OpenAI enhancement

import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
import warnings
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

# Optional OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("OpenAI API key not found in environment variables (OPENAI_API_KEY).")
        OPENAI_AVAILABLE = False
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Using basic NLP only.")

class PropertyNLP:
    """
    Simplified NLP processor for real estate descriptions
    """

    def __init__(self, use_openai=False):
        """Initialize with optional OpenAI support"""
        self.use_openai = use_openai and OPENAI_AVAILABLE
        if self.use_openai:
            print("OpenAI integration enabled.")
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Expanded keyword lists
        self.luxury_words = ['luxury', 'premium', 'elegant', 'stunning', 'beautiful', 'custom', 'designer', 'gourmet', 'exclusive', 'high-end']
        self.condition_words = ['new', 'renovated', 'updated', 'modern', 'fresh', 'recent', 'remodeled', 'immaculate']
        self.amenity_words = ['pool', 'spa', 'garage', 'fireplace', 'patio', 'deck', 'garden', 'view', 'waterfront', 'gated']

    def _preprocess_text(self, text):
        """Clean and lemmatize text."""
        if pd.isna(text):
            return []
        text = str(text).lower()
        text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
        tokens = nltk.word_tokenize(text)
        lemmas = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        return lemmas

    def analyze_with_openai(self, description):
        """Use OpenAI to analyze property description"""
        if not self.use_openai or pd.isna(description):
            return {'openai_sentiment': np.nan, 'openai_luxury_score': np.nan, 'openai_appeal': np.nan}
        
        try:
            prompt = f"""
            Analyze this real estate property description and return ONLY a JSON object with these exact keys:
            - "sentiment": a float from -1.0 (negative) to 1.0 (positive).
            - "luxury_score": an integer from 0 to 10 (how luxurious/high-end).
            - "appeal": an integer from 0 to 10 (overall marketing appeal).
            
            Description: {description[:1000]}
            
            JSON:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            result = eval(response.choices[0].message.content.strip())
            return {
                'openai_sentiment': result.get('sentiment', 0),
                'openai_luxury_score': result.get('luxury_score', 0),
                'openai_appeal': result.get('appeal', 0)
            }
        except Exception as e:
            print(f"OpenAI analysis failed: {e}")
            return {'openai_sentiment': np.nan, 'openai_luxury_score': np.nan, 'openai_appeal': np.nan}
    
    def extract_basic_features(self, description):
        """Extract basic NLP features"""
        if pd.isna(description):
            description = ""
        
        lemmas = self._preprocess_text(description)
        
        sentiment = TextBlob(description).sentiment.polarity
        
        luxury_count = sum(1 for word in self.luxury_words if word in lemmas)
        condition_count = sum(1 for word in self.condition_words if word in lemmas)
        amenity_count = sum(1 for word in self.amenity_words if word in lemmas)
        
        features = {
            'description_length': len(description),
            'sentiment': sentiment,
            'luxury_keyword_count': luxury_count,
            'condition_keyword_count': condition_count,
            'amenity_keyword_count': amenity_count,
            'unique_word_count': len(set(lemmas)),
            'exclamation_count': description.count('!')
        }
        
        if self.use_openai:
            openai_features = self.analyze_with_openai(description)
            features.update(openai_features)
        
        return features
    
    def process_descriptions(self, descriptions):
        """Process a series of descriptions and return a DataFrame of features."""
        features_list = []
        total = len(descriptions)
        
        for i, desc in enumerate(descriptions):
            if (i + 1) % 100 == 0:
                print(f"Processing description {i+1}/{total}...")
            
            features = self.extract_basic_features(desc)
            features_list.append(features)
        
        return pd.DataFrame(features_list)

def analyze_nlp_features(use_openai=False, sample_size=1000):
    """
    Load data, analyze NLP features, and show correlations with price ranges.
    """
    print("=" * 50)
    print("NLP FEATURE ANALYSIS")
    print("=" * 50)
    
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Error: data.csv not found. Make sure you are in the correct directory.")
        return None, None

    if sample_size < len(df):
        sample_df = df.sample(n=sample_size, random_state=42)
        print(f"Analyzing {sample_size} random properties...")
    else:
        sample_df = df
        print(f"Analyzing all {len(df)} properties...")
    
    print("\nExtracting NLP features...")
    nlp_processor = PropertyNLP(use_openai=use_openai)
    nlp_features = nlp_processor.process_descriptions(sample_df['description'])
    
    nlp_features['priceRange'] = sample_df['priceRange'].values
    
    price_mapping = {'0-250000': 1, '250000-350000': 2, '350000-450000': 3, 
                     '450000-650000': 4, '650000+': 5}
    nlp_features['price_numeric'] = nlp_features['priceRange'].map(price_mapping)
    
    print(f"\nNLP Feature Correlations with Price Range:")
    print("-" * 50)
    
    feature_cols = [col for col in nlp_features.columns if 'price' not in col]
    correlations = nlp_features[feature_cols].corrwith(nlp_features['price_numeric']).dropna()
    
    for feature, corr in correlations.sort_values(ascending=False).items():
        print(f"{feature:25s}: {corr:6.3f}")
    
    print(f"\nNLP Features by Price Range (Mean):")
    print("-" * 50)
    
    summary = nlp_features.groupby('priceRange')[feature_cols].mean()
    print(summary.to_string(float_format="%.2f"))
    
    # Save results
    output_file = 'nlp_analysis_results.csv'
    # Combine original data with NLP features for context
    result_df = pd.concat([sample_df.reset_index(drop=True), nlp_features], axis=1)
    result_df.to_csv(output_file, index=False)
    print(f"\nFull analysis results saved to: {output_file}")
    
    return nlp_features, correlations

def main():
    """
    Main NLP analysis function
    """
    print("=" * 50)
    print("REAL ESTATE NLP ANALYSIS")
    print("=" * 50)
    
    use_openai_input = input("Use OpenAI for enhanced analysis? (Requires OPENAI_API_KEY env var) (y/n): ").lower().strip()
    use_openai = use_openai_input == 'y'

    if use_openai and not os.getenv("OPENAI_API_KEY"):
        print("OpenAI requested, but API key is not set. Proceeding with basic analysis.")
        use_openai = False
    
    sample_size_input = input("Enter sample size to analyze (or press Enter for 1000): ").strip()
    try:
        sample_size = int(sample_size_input) if sample_size_input else 1000
    except ValueError:
        print("Invalid number, defaulting to 1000.")
        sample_size = 1000
    
    analyze_nlp_features(use_openai=use_openai, sample_size=sample_size)
    
    print(f"\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main() 