# Real Estate Price Range Prediction Model
# This script creates a machine learning model to predict property price ranges using XGBoost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and perform initial exploration of the dataset"""
    print("=" * 60)
    print("REAL ESTATE PRICE PREDICTION MODEL - XGBoost")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataset Info:")
    df.info()
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nPrice Range Distribution:")
    price_counts = df['priceRange'].value_counts()
    print(price_counts)
    
    return df

def clean_and_engineer_features(df):
    """Clean data and create new features"""
    print("\n" + "=" * 40)
    print("DATA CLEANING AND FEATURE ENGINEERING")
    print("=" * 40)
    
    df_processed = df.copy()
    
    print("Handling missing values...")
    
    numerical_cols = ['lotSizeSqFt', 'avgSchoolRating', 'MedianStudentsPerTeacher', 
                      'numOfBathrooms', 'numOfBedrooms', 'yearBuilt', 'garageSpaces', 
                      'numOfPatioAndPorchFeatures']
    
    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    categorical_cols = ['city', 'homeType', 'hasSpa']
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    print("Creating new features...")
    current_year = 2024
    df_processed['property_age'] = current_year - df_processed['yearBuilt']
    df_processed['total_rooms'] = df_processed['numOfBathrooms'] + df_processed['numOfBedrooms']
    df_processed['bath_bed_ratio'] = df_processed['numOfBathrooms'] / (df_processed['numOfBedrooms'] + 1)
    df_processed['lot_per_room'] = df_processed['lotSizeSqFt'] / (df_processed['total_rooms'] + 1)
    df_processed['hasSpa'] = df_processed['hasSpa'].astype(int)
    
    print("Missing values after cleaning:")
    print(df_processed.isnull().sum().sum())
    
    return df_processed

def visualize_data(df):
    """Create visualizations for data exploration"""
    print("\n" + "=" * 30)
    print("DATA VISUALIZATION")
    print("=" * 30)
    
    # Price range distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    price_counts = df['priceRange'].value_counts()
    price_counts.plot(kind='bar')
    plt.title('Price Range Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.pie(price_counts.values, labels=price_counts.index, autopct='%1.1f%%')
    plt.title('Price Range Percentage')
    
    # Geographic distribution
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                         c=pd.Categorical(df['priceRange']).codes, 
                         alpha=0.6, s=1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    # Feature distributions by price range
    plt.figure(figsize=(18, 12))
    key_features = ['numOfBedrooms', 'numOfBathrooms', 'lotSizeSqFt', 'avgSchoolRating', 'property_age', 'total_rooms']
    
    for i, feature in enumerate(key_features):
        plt.subplot(2, 3, i+1)
        df.boxplot(column=feature, by='priceRange', ax=plt.gca())
        plt.title(f'{feature} by Price Range')
        plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    plt.show()

def prepare_ml_data(df):
    """Prepare data for machine learning"""
    print("\n" + "=" * 40)
    print("PREPARING DATA FOR MACHINE LEARNING")
    print("=" * 40)
    
    features_to_drop = ['uid', 'description']
    df_ml = df.drop(columns=features_to_drop)
    
    X = df_ml.drop('priceRange', axis=1)
    y = df_ml['priceRange']
    
    categorical_features = ['city', 'homeType']
    numerical_features = ['latitude', 'longitude', 'garageSpaces', 'hasSpa', 'yearBuilt',
                         'numOfPatioAndPorchFeatures', 'lotSizeSqFt', 'avgSchoolRating',
                         'MedianStudentsPerTeacher', 'numOfBathrooms', 'numOfBedrooms',
                         'property_age', 'total_rooms', 'bath_bed_ratio', 'lot_per_room']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print(f"Features for ML: {X.columns.tolist()}")
    print(f"Target classes: {y.unique()}")
    
    return X, y, preprocessor, categorical_features, numerical_features

def train_xgboost_model(X, y, preprocessor):
    """Train XGBoost model and evaluate performance"""
    print("\n" + "=" * 30)
    print("XGBOOST MODEL TRAINING")
    print("=" * 30)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Create XGBoost model with optimized parameters
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])
    
    print("Training XGBoost model...")
    pipeline.fit(X_train, y_train)
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    # Test predictions
    print("Making predictions on test set...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return pipeline, X_test, y_test, y_pred, y_pred_proba, cv_scores

def analyze_xgboost_model(model, X_test, y_test, y_pred, y_pred_proba, cv_scores):
    """Analyze the XGBoost model performance in detail"""
    print("\n" + "=" * 30)
    print("DETAILED MODEL ANALYSIS")
    print("=" * 30)
    
    # Performance metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Final Model: XGBoost Classifier")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"✅ CV Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"✅ CV Std: {cv_scores.std():.4f}")
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confidence analysis
    max_probabilities = np.max(y_pred_proba, axis=1)
    avg_confidence = np.mean(max_probabilities)
    print(f"\n🎯 Average Prediction Confidence: {avg_confidence:.4f} ({avg_confidence:.1%})")
    
    # Confidence by prediction
    confidence_by_class = {}
    for i, class_name in enumerate(model.classes_):
        mask = y_pred == class_name
        if np.any(mask):
            class_confidence = np.mean(max_probabilities[mask])
            confidence_by_class[class_name] = class_confidence
            print(f"   {class_name}: {class_confidence:.3f}")
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Feature Importance
    plt.subplot(2, 3, 2)
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = (model.named_steps['preprocessor']
                        .named_transformers_['num'].get_feature_names_out().tolist() +
                        model.named_steps['preprocessor']
                        .named_transformers_['cat'].get_feature_names_out().tolist())
        
        importances = model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()
    
    # Accuracy by Price Range
    plt.subplot(2, 3, 5)
    accuracy_by_class = []
    class_names = []
    for class_name in model.classes_:
        mask = y_test == class_name
        if np.any(mask):
            class_accuracy = accuracy_score(y_test[mask], y_pred[mask])
            accuracy_by_class.append(class_accuracy)
            class_names.append(class_name)
    
    plt.bar(range(len(class_names)), accuracy_by_class)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.title('Accuracy by Price Range')
    plt.ylabel('Accuracy')
    
    #Prediction vs Actual scatterplot
    plt.subplot(2, 3, 6)
    price_mapping = {price: i for i, price in enumerate(model.classes_)}
    y_test_numeric = [price_mapping[price] for price in y_test]
    y_pred_numeric = [price_mapping[price] for price in y_pred]
    
    plt.scatter(y_test_numeric, y_pred_numeric, alpha=0.6)
    plt.plot([0, len(model.classes_)-1], [0, len(model.classes_)-1], 'r--')
    plt.xlabel('Actual Price Range')
    plt.ylabel('Predicted Price Range')
    plt.title('Predictions vs Actual')
    plt.xticks(range(len(model.classes_)), model.classes_, rotation=45)
    plt.yticks(range(len(model.classes_)), model.classes_, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return test_accuracy, avg_confidence

def create_prediction_function(model):
    """Create a function for making predictions on new data"""
    def predict_price_range(property_data):
        """
        Predict price range for a new property
        
        Parameters:
        property_data: dict with property features
        
        Returns:
        prediction and probabilities
        """
        property_df = pd.DataFrame([property_data])
        
        current_year = 2024
        property_df['property_age'] = current_year - property_df['yearBuilt']
        property_df['total_rooms'] = property_df['numOfBathrooms'] + property_df['numOfBedrooms']
        property_df['bath_bed_ratio'] = property_df['numOfBathrooms'] / (property_df['numOfBedrooms'] + 1)
        property_df['lot_per_room'] = property_df['lotSizeSqFt'] / (property_df['total_rooms'] + 1)
        property_df['hasSpa'] = property_df['hasSpa'].astype(int)
        
        prediction = model.predict(property_df)[0]
        probabilities = model.predict_proba(property_df)[0]
        
        prob_dict = dict(zip(model.classes_, probabilities))
        
        return prediction, prob_dict
    
    return predict_price_range

def save_model(model, categorical_features, numerical_features):
    """Save the trained model"""
    print("\n" + "=" * 20)
    print("SAVING MODEL")
    print("=" * 20)
    
    model_filename = 'best_price_prediction_model_xgboost.pkl'
    joblib.dump(model, model_filename)
    
    print(f"✅ Model saved as: {model_filename}")
    
    return model_filename

def main():
    """Main function to run the complete pipeline"""
    df = load_and_explore_data()
    
    df_processed = clean_and_engineer_features(df)
    
    visualize_data(df_processed)
    
    X, y, preprocessor, categorical_features, numerical_features = prepare_ml_data(df_processed)
    
    model, X_test, y_test, y_pred, y_pred_proba, cv_scores = train_xgboost_model(X, y, preprocessor)
    
    test_accuracy, avg_confidence = analyze_xgboost_model(model, X_test, y_test, y_pred, y_pred_proba, cv_scores)
    
    predict_price_range = create_prediction_function(model)
    
    # 9. Save model
    model_filename = save_model(model, categorical_features, numerical_features)
    
    print("\n" + "=" * 60)
    print("🎉 XGBOOST MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"✅ Model: XGBoost Classifier")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"✅ Average Confidence: {avg_confidence:.4f}")
    print(f"✅ Model saved as: {model_filename}")
    print("=" * 60)
    
    return model, predict_price_range

if __name__ == "__main__":
    model, predict_price_range = main() 