# Real Estate Price Prediction Model Deployment
# Production-ready model deployment with testing capabilities

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import accuracy_score
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class RealEstatePricePredictor:
    """
    Production-ready Real Estate Price Range Predictor
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor with model file
        
        Args:
            model_path: Path to the trained model pickle file
        """
        self.model = None
        self.is_loaded = False
        
        # Default path
        if model_path is None:
            model_path = 'best_price_prediction_random_forest.pkl'
            
        self.model_path = model_path
        
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except FileNotFoundError as e:
            print(f"Error: Model file not found - {e}")
            self.is_loaded = False
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def _validate_input(self, property_data: Dict) -> Tuple[bool, str]:
        """
        Validate input property data
        
        Args:
            property_data: Dictionary containing property features
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_features = [
            'city', 'homeType', 'latitude', 'longitude', 'garageSpaces',
            'hasSpa', 'yearBuilt', 'numOfPatioAndPorchFeatures', 'lotSizeSqFt',
            'avgSchoolRating', 'MedianStudentsPerTeacher', 'numOfBathrooms', 'numOfBedrooms'
        ]
        
        missing_features = [f for f in required_features if f not in property_data]
        if missing_features:
            return False, f"Missing required features: {missing_features}"
        
        try:
            if not isinstance(property_data['latitude'], (int, float)) or not (-90 <= property_data['latitude'] <= 90):
                return False, "Invalid latitude: must be between -90 and 90"
            
            if not isinstance(property_data['longitude'], (int, float)) or not (-180 <= property_data['longitude'] <= 180):
                return False, "Invalid longitude: must be between -180 and 180"
            
            if not isinstance(property_data['yearBuilt'], int) or not (1800 <= property_data['yearBuilt'] <= 2024):
                return False, "Invalid yearBuilt: must be between 1800 and 2024"
            
            if not isinstance(property_data['numOfBedrooms'], (int, float)) or property_data['numOfBedrooms'] < 0:
                return False, "Invalid numOfBedrooms: must be non-negative"
            
            if not isinstance(property_data['numOfBathrooms'], (int, float)) or property_data['numOfBathrooms'] < 0:
                return False, "Invalid numOfBathrooms: must be non-negative"
            
            if not isinstance(property_data['lotSizeSqFt'], (int, float)) or property_data['lotSizeSqFt'] < 0:
                return False, "Invalid lotSizeSqFt: must be non-negative"
            
            if not isinstance(property_data['avgSchoolRating'], (int, float)) or not (0 <= property_data['avgSchoolRating'] <= 10):
                return False, "Invalid avgSchoolRating: must be between 0 and 10"
            
        except Exception as e:
            return False, f"Data validation error: {e}"
        
        return True, ""
    
    def _engineer_features(self, property_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for the property data
        
        Args:
            property_df: DataFrame with property features
            
        Returns:
            DataFrame with engineered features added
        """
        current_year = 2024
        property_df = property_df.copy()
        
        property_df['property_age'] = current_year - property_df['yearBuilt']
        property_df['total_rooms'] = property_df['numOfBathrooms'] + property_df['numOfBedrooms']
        property_df['bath_bed_ratio'] = property_df['numOfBathrooms'] / (property_df['numOfBedrooms'] + 1)
        property_df['lot_per_room'] = property_df['lotSizeSqFt'] / (property_df['total_rooms'] + 1)
        property_df['hasSpa'] = property_df['hasSpa'].astype(int)
        
        return property_df
    
    def predict_single(self, property_data: Dict) -> Dict[str, Any]:
        """
        Predict price range for a single property
        
        Args:
            property_data: Dictionary containing property features
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Model not loaded. Please check model file.',
                'prediction': None,
                'probabilities': None,
                'confidence': None
            }
        
        is_valid, error_msg = self._validate_input(property_data)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg,
                'prediction': None,
                'probabilities': None,
                'confidence': None
            }
        
        try:
            property_df = pd.DataFrame([property_data])
            
            property_df = self._engineer_features(property_df)
            
            prediction = self.model.predict(property_df)[0]
            probabilities = self.model.predict_proba(property_df)[0]
            
            prob_dict = dict(zip(self.model.classes_, probabilities))
            confidence = max(probabilities)
            
            return {
                'success': True,
                'error': None,
                'prediction': prediction,
                'probabilities': prob_dict,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction error: {str(e)}',
                'prediction': None,
                'probabilities': None,
                'confidence': None
            }
    
    def predict_batch(self, properties_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        Predict price ranges for multiple properties
        
        Args:
            properties_list: List of dictionaries containing property features
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for i, property_data in enumerate(properties_list):
            result = self.predict_single(property_data)
            result['property_index'] = i
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        return {
            'model_name': 'Random Forest',
            'model_type': 'RandomForestClassifier',
            'price_classes': list(self.model.classes_),
            'model_loaded': self.is_loaded,
            'model_path': self.model_path,
            'required_features': [
                'city', 'homeType', 'latitude', 'longitude', 'garageSpaces',
                'hasSpa', 'yearBuilt', 'numOfPatioAndPorchFeatures', 'lotSizeSqFt',
                'avgSchoolRating', 'MedianStudentsPerTeacher', 'numOfBathrooms', 'numOfBedrooms'
            ]
        }

def test_against_training_data(predictor: RealEstatePricePredictor, 
                             data_path: str = 'data.csv',
                             sample_size: int = 100) -> Dict[str, Any]:
    """
    Test the deployed model against the training data
    
    Args:
        predictor: Initialized RealEstatePricePredictor instance
        data_path: Path to the training data CSV file
        sample_size: Number of samples to test (default 100)
        
    Returns:
        Dictionary with test results
    """
    print("=" * 60)
    print("TESTING MODEL AGAINST TRAINING DATA")
    print("=" * 60)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded training data: {df.shape[0]} rows")
        
        if sample_size < len(df):
            test_df = df.sample(n=sample_size, random_state=42)
            print(f"Testing on {sample_size} random samples")
        else:
            test_df = df
            print(f"Testing on all {len(df)} samples")
        
        test_results = []
        actual_values = []
        predicted_values = []
        
        print("\nRunning predictions...")
        for idx, row in test_df.iterrows():
            #Prepare property data (exclude uid, description, and priceRange)
            property_data = {
                'city': row['city'],
                'homeType': row['homeType'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'garageSpaces': row['garageSpaces'],
                'hasSpa': int(row['hasSpa']) if isinstance(row['hasSpa'], bool) else row['hasSpa'],
                'yearBuilt': row['yearBuilt'],
                'numOfPatioAndPorchFeatures': row['numOfPatioAndPorchFeatures'],
                'lotSizeSqFt': row['lotSizeSqFt'],
                'avgSchoolRating': row['avgSchoolRating'],
                'MedianStudentsPerTeacher': row['MedianStudentsPerTeacher'],
                'numOfBathrooms': row['numOfBathrooms'],
                'numOfBedrooms': row['numOfBedrooms']
            }
            
            result = predictor.predict_single(property_data)
            
            if result['success']:
                actual_price = row['priceRange']
                predicted_price = result['prediction']
                
                actual_values.append(actual_price)
                predicted_values.append(predicted_price)
                
                test_results.append({
                    'index': idx,
                    'actual': actual_price,
                    'predicted': predicted_price,
                    'correct': actual_price == predicted_price,
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
            else:
                print(f"Error predicting for row {idx}: {result['error']}")
        
        accuracy = accuracy_score(actual_values, predicted_values)
        
        results_df = pd.DataFrame(test_results)
        price_range_accuracy = results_df.groupby('actual').agg({
            'correct': ['count', 'sum', 'mean']
        }).round(4)
        
        print(f"\nTest Results:")
        print(f"Total predictions: {len(predicted_values)}")
        print(f"Correct predictions: {sum(results_df['correct'])}")
        print(f"Overall accuracy: {accuracy:.4f} ({accuracy:.1%})")
        
        print(f"\nAccuracy by Price Range:")
        print(price_range_accuracy)
        
        avg_confidence = results_df['confidence'].mean()
        print(f"\nAverage prediction confidence: {avg_confidence:.4f} ({avg_confidence:.1%})")
        
        return {
            'total_predictions': len(predicted_values),
            'correct_predictions': sum(results_df['correct']),
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'results_by_price_range': price_range_accuracy.to_dict(),
            'detailed_results': test_results
        }
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    predictor = RealEstatePricePredictor()
    
    if predictor.is_loaded:
        test_results = test_against_training_data(predictor, sample_size=200)
        print("\nDeployment testing completed!")
    else:
        print("Error: Could not load model. Please run the training script first.") 