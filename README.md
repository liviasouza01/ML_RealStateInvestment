# Real Estate Price Range Prediction Model

A comprehensive machine learning solution for predicting real estate property price ranges based on property characteristics using XGBoost.

## Overview

This project implements a complete machine learning pipeline that:
- Trains an XGBoost classification model on real estate data.
- Provides a production-ready deployment class with comprehensive testing.
- Includes a module for Natural Language Processing (NLP) analysis on property descriptions.

## Files Structure

```
ML_RealStateInvestment/
├── data.csv                                    # Training dataset
├── requirements.txt                            # Python dependencies
├── model_training.py                           # Main model training script (XGBoost)
├── model_deployment.py                         # Production deployment and testing
├── nlp_implementation.py                       # NLP analysis of property descriptions
├── best_price_prediction_model_xgboost.pkl     # Trained XGBoost model (generated)
└── README.md                                   # This file
```

## Quick Start with a Virtual Environment

### 1. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS and Linux:
source .venv/bin/activate
# On Windows:
# .\.venv\Scripts\activate
```

### 2. Install Dependencies

Once your virtual environment is activated, install the required Python packages.

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional - Model Already Provided)

This script trains the XGBoost model and saves it as `best_price_prediction_model_xgboost.pkl`.

```bash
python model_training.py
```

### 4. Test the Deployed Model

This script tests the trained model against a sample of the training data to verify its performance.

```bash
python model_deployment.py
```

### 5. Run NLP Analysis on Property Descriptions

This script analyzes the text descriptions from `data.csv` to find correlations with price. It can optionally use the OpenAI API for enhanced analysis.

```bash
python nlp_implementation.py
```

## Usage Example

### Making a Prediction

Use the `RealEstatePricePredictor` class from `model_deployment.py` for predictions.

```python
from model_deployment import RealEstatePricePredictor

# Initialize predictor (loads the .pkl model)
predictor = RealEstatePricePredictor()

# Example property data
property_data = {
    'city': 'austin',
    'homeType': 'Single Family',
    'latitude': 30.2672,
    'longitude': -97.7431,
    'garageSpaces': 2,
    'hasSpa': 1,
    'yearBuilt': 2010,
    'numOfPatioAndPorchFeatures': 2,
    'lotSizeSqFt': 10000.0,
    'avgSchoolRating': 8.5,
    'MedianStudentsPerTeacher': 15,
    'numOfBathrooms': 3.5,
    'numOfBedrooms': 4
}

# Make prediction
result = predictor.predict_single(property_data)

if result['success']:
    print(f"Predicted Price Range: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
else:
    print(f"Error: {result['error']}")
```

## Required Features

All predictions require the following property features:

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| city | string | City name | e.g., "austin" |
| homeType | string | Property type | "Single Family", "Condo", etc. |
| latitude | float | Property latitude | -90 to 90 |
| longitude | float | Property longitude | -180 to 180 |
| garageSpaces | int | Number of garage spaces | 0+ |
| hasSpa | int | Has spa (boolean) | 0 or 1 |
| yearBuilt | int | Year built | 1800-2024 |
| numOfPatioAndPorchFeatures | int | Patio/porch features | 0+ |
| lotSizeSqFt | float | Lot size (sq ft) | 0+ |
| avgSchoolRating | float | Average school rating | 0-10 |
| MedianStudentsPerTeacher | int | Student-teacher ratio | 1+ |
| numOfBathrooms | float | Number of bathrooms | 0+ |
| numOfBedrooms | int | Number of bedrooms | 0+ |

## Price Range Categories

The model predicts one of five price ranges:

1. **0-250000**: $0 - $250,000
2. **250000-350000**: $250,000 - $350,000
3. **350000-450000**: $350,000 - $450,000
4. **450000-650000**: $450,000 - $650,000
5. **650000+**: $650,000+

## Model Details

### Engineered Features
The model automatically creates additional features for training and prediction:
- `property_age`: Current year - year built
- `total_rooms`: Bathrooms + bedrooms
- `bath_bed_ratio`: Bathrooms / (bedrooms + 1)
- `lot_per_room`: Lot size / (total rooms + 1)

### Model Pipeline
- **Preprocessing**: `StandardScaler` for numerical features, `OneHotEncoder` for categorical.
- **Algorithm**: `XGBClassifier` with optimized hyperparameters.
- **Validation**: 5-fold cross-validation during training.

## NLP Analysis

The `nlp_implementation.py` script performs analysis on property descriptions to identify how language correlates with price.
- It calculates sentiment, keyword counts, and other text-based features.
- It can optionally use the OpenAI API for more advanced analysis.
- The script **does not** train a model; it only provides insights into the text data.
- Results of the analysis are saved to `nlp_analysis_results.csv`.

## Production Deployment

The `RealEstatePricePredictor` class in `model_deployment.py` is designed for production use and includes:
- Input validation and clear error handling.
- Batch prediction capabilities for efficiency.
- A simple interface for loading the model and making predictions.

## Testing

The deployment includes a built-in testing function that validates the model against the training data.

```python
from model_deployment import test_against_training_data, RealEstatePricePredictor

predictor = RealEstatePricePredictor()
results = test_against_training_data(predictor, sample_size=200)
```

### Test Results (on 200 samples)
- **Overall Accuracy**: ~67.0%
- **Average Confidence**: ~57%
