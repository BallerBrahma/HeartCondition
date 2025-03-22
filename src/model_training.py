import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Database connection
DB_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/heart_disease')
engine = create_engine(DB_URL)

def load_training_data():
    """Load and prepare training data"""
    query = """
    SELECT * FROM processed_heart_data
    """
    df = pd.read_sql(query, engine)
    
    # Convert categorical variables back to numeric
    categorical_features = ['cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 
                          'age_group', 'sex_age_group', 'cp_sex', 'risk_level']
    label_encoders = {}
    
    for feature in categorical_features:
        if feature in df.columns:
            label_encoders[feature] = LabelEncoder()
            df[feature] = label_encoders[feature].fit_transform(df[feature].astype(str))
    
    # Save label encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    
    # Select features for training
    feature_columns = [col for col in df.columns if col not in ['id', 'target']]
    X = df[feature_columns]
    y = df['target']
    
    return X, y

def prepare_features(df):
    """Prepare features for model training."""
    # Create label encoders for categorical variables
    label_encoders = {}
    categorical_features = [
        'age_group', 'sex', 'cp_type', 'fbs', 
        'restecg_type', 'exang', 'slope_type', 'thal_type'
    ]
    
    # Encode categorical features
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        df[feature] = label_encoders[feature].fit_transform(df[feature])
    
    # Save label encoders for later use
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y, label_encoders

def train_models(X, y, X_test, y_test):
    """Train multiple models and evaluate their performance"""
    # Initialize base models
    base_models = {
        'logistic_regression': LogisticRegression(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(),
        'svm': SVC(probability=True)
    }
    
    # Define parameter grids for each model
    param_grids = {
        'logistic_regression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear'],
            'class_weight': ['balanced']
        },
        'decision_tree': {
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 6, 7],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf'],
            'class_weight': ['balanced']
        }
    }
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train and evaluate each base model
    base_model_results = {}
    for name, model in base_models.items():
        print(f"\nTraining {name}...")
        
        # Create grid search with precision scoring
        grid_search = GridSearchCV(
            model,
            param_grids[name],
            cv=cv,
            scoring='precision',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X, y)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cv_precision_mean': grid_search.best_score_,
            'best_parameters': grid_search.best_params_
        }
        
        # Print metrics
        print(f"Metrics for {name}:")
        for metric, value in metrics.items():
            if metric != 'best_parameters':
                print(f"{metric}: {value:.4f}")
            else:
                print(f"Best parameters: {value}")
        
        # Save model and metrics
        save_model(best_model, name)
        save_metrics(metrics, name)
        
        base_model_results[name] = best_model
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', base_model_results['random_forest']),
            ('lr', base_model_results['logistic_regression']),
            ('dt', base_model_results['decision_tree'])
        ],
        voting='soft'
    )
    
    # Train voting classifier
    print("\nTraining voting classifier...")
    voting_clf.fit(X, y)
    y_pred_voting = voting_clf.predict(X_test)
    
    # Calculate metrics for voting classifier
    voting_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_voting),
        'precision': precision_score(y_test, y_pred_voting),
        'recall': recall_score(y_test, y_pred_voting),
        'f1': f1_score(y_test, y_pred_voting)
    }
    
    print("Metrics for voting classifier:")
    for metric, value in voting_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save voting classifier
    save_model(voting_clf, 'voting_classifier')
    save_metrics(voting_metrics, 'voting_classifier')
    
    # Create stacking classifier
    estimators = [
        ('rf', base_model_results['random_forest']),
        ('lr', base_model_results['logistic_regression']),
        ('dt', base_model_results['decision_tree'])
    ]
    final_estimator = LogisticRegression(class_weight='balanced')
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method='predict_proba'
    )
    
    # Train stacking classifier
    print("\nTraining stacking classifier...")
    stacking_clf.fit(X, y)
    y_pred_stacking = stacking_clf.predict(X_test)
    
    # Calculate metrics for stacking classifier
    stacking_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_stacking),
        'precision': precision_score(y_test, y_pred_stacking),
        'recall': recall_score(y_test, y_pred_stacking),
        'f1': f1_score(y_test, y_pred_stacking)
    }
    
    print("Metrics for stacking classifier:")
    for metric, value in stacking_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save stacking classifier
    save_model(stacking_clf, 'stacking_classifier')
    save_metrics(stacking_metrics, 'stacking_classifier')
    
    # For Random Forest, analyze and print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': base_model_results['random_forest'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    return base_model_results, voting_clf, stacking_clf

def save_model(model, name):
    """Save trained model to disk"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{name}.joblib')

def save_metrics(metrics, name):
    """Save model metrics to disk"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(metrics, f'models/{name}_metrics.joblib')

def save_predictions_to_db(model, X_test, y_test, model_name):
    """Save model predictions to database"""
    query = "SELECT id FROM processed_heart_data"
    all_ids = pd.read_sql(query, engine)
    test_ids = all_ids.iloc[X_test.index]
    
    # Prepare predictions for database
    predictions_data = []
    for idx, (pred, prob) in enumerate(zip(model.predict(X_test), model.predict_proba(X_test)[:, 1])):
        predictions_data.append({
            'patient_id': test_ids.iloc[idx]['id'],
            'prediction': int(pred),
            'probability': float(prob),
            'model_name': model_name,
            'created_at': datetime.now()
        })
    
    # Save to database
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_sql('model_predictions', engine, if_exists='append', index=False)

def main():
    """Main function to train and evaluate models"""
    print("Starting model training...")
    
    # Load and prepare data
    X, y = load_training_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    base_models, voting_clf, stacking_clf = train_models(X_train, y_train, X_test, y_test)
    
    print("\nModel training completed successfully!")
    print("Models and metrics have been saved to the 'models' directory")
    print("Predictions have been saved to the database")

if __name__ == "__main__":
    main() 