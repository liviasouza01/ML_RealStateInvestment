# Real Estate Price Range Prediction Model
# This script creates a machine learning model to predict property price ranges using Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and perform initial exploration of the dataset"""
    print("=" * 60)
    print("REAL ESTATE PRICE PREDICTION MODEL - Random Forest")
    print("=" * 60)
    
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
                      'numOfPatioAndPorchFeatures', 'latitude', 'longitude']
    
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
    """Create visualizations for data exploration and save them to disk."""
    print("\n" + "=" * 30)
    print("DATA VISUALIZATION")
    print("=" * 30)
    
    # Ensure plots directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    #Price range distribution
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
    
    #Geographic distribution
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                         c=pd.Categorical(df['priceRange']).codes, 
                         alpha=0.6, s=1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    
    plot_path = 'plots/data_visualization.png'
    plt.savefig(plot_path)
    print(f"Data visualization plot saved to {plot_path}")

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
    
    return X, y, preprocessor, categorical_features, numerical_features

def train_model(X, y, preprocessor):
    """Train Random Forest model with hyperparameter tuning and evaluate performance"""
    print("\n" + "=" * 30)
    print("RANDOM FOREST MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 30)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # Define the parameter space for RandomizedSearchCV
    param_dist = {
        'classifier__n_estimators': [100, 200, 300, 400],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__bootstrap': [True, False]
    }

    print("Performing Randomized Search with Cross-Validation...")
    print("This may take a few minutes...")
    
    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist,
        n_iter=25,  # Number of parameter settings that are sampled
        cv=3,       # 3-fold cross-validation
        verbose=1, 
        random_state=42, 
        n_jobs=-1,
        scoring='accuracy'
    )
    
    random_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    print(random_search.best_params_)
    
    best_model = random_search.best_estimator_
    
    print("\nMaking predictions on test set with the best model...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Re-evaluate CV scores on the best model for a consistent final metric
    print("\nPerforming final cross-validation on the best model...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')

    print(f"\n📊 BEST MODEL PERFORMANCE:")
    print(f"Final Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    return best_model, X_test, y_test, y_pred, y_pred_proba, cv_scores

def analyze_model_performance(model, X_test, y_test, y_pred, y_pred_proba, cv_scores):
    """Analyze the model performance in detail and save plots."""
    print("\n" + "=" * 30)
    print("DETAILED MODEL ANALYSIS")
    print("=" * 30)
    
    if not os.path.exists('plots'):
        os.makedirs('plots')

    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    weighted_avg = report.get('weighted avg', {})
    precision = weighted_avg.get('precision', 0)
    recall = weighted_avg.get('recall', 0)
    f1 = weighted_avg.get('f1-score', 0)

    print(f"Model: Random Forest Classifier")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print(f"CV Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"CV Std: {cv_scores.std():.4f}")
    
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    max_probabilities = np.max(y_pred_proba, axis=1)
    avg_confidence = np.mean(max_probabilities)
    print(f"\nAverage Prediction Confidence: {avg_confidence:.4f} ({avg_confidence:.1%})")
    
    confidence_by_class = {}
    for i, class_name in enumerate(model.classes_):
        mask = y_pred == class_name
        if np.any(mask):
            class_confidence = np.mean(max_probabilities[mask])
            confidence_by_class[class_name] = class_confidence
            print(f"   {class_name}: {class_confidence:.3f}")
    
    plt.figure(figsize=(20, 8))

    # Feature Importance
    plt.subplot(1, 3, 2)
    try:
        # Get feature names from the preprocessor
        cat_features = model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()
        num_features = model.named_steps['preprocessor'].transformers_[0][2]
        all_features = list(num_features) + list(cat_features)
        
        importances = model.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_features, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(15)
        
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('Top 15 Feature Importances')
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

    #Prediction vs Actual scatterplot
    plt.subplot(1, 3, 3)
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
    
    plot_path = 'plots/model_performance_analysis.png'
    plt.savefig(plot_path)
    print(f"Model performance plot saved to {plot_path}")

    plt.show()
    
    return test_accuracy, avg_confidence, precision, recall, f1

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

def save_model(model):
    """Save the trained model"""
    print("\n" + "=" * 20)
    print("SAVING MODEL")
    print("=" * 20)
    
    model_filename = 'best_price_prediction_random_forest.pkl'
    joblib.dump(model, model_filename)
    
    print(f"Model saved as: {model_filename}")
    
    return model_filename

def main():
    """Main function to run the complete pipeline"""
    df = load_and_explore_data()
    
    df_processed = clean_and_engineer_features(df)
    
    visualize_data(df_processed)
    
    X, y, preprocessor, categorical_features, numerical_features = prepare_ml_data(df_processed)
    
    model, X_test, y_test, y_pred, y_pred_proba, cv_scores = train_model(X, y, preprocessor)
    
    test_accuracy, avg_confidence, precision, recall, f1 = analyze_model_performance(model, X_test, y_test, y_pred, y_pred_proba, cv_scores)
    
    predict_price_range = create_prediction_function(model)
    
    model_filename = save_model(model)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model: Random Forest Classifier")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Model saved as: {model_filename}")
    print("=" * 60)
    
    return model, predict_price_range

if __name__ == "__main__":
    model, predict_price_range = main() 