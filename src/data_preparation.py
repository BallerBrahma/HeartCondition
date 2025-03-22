import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import os
from dotenv import load_dotenv
from sqlalchemy import text

# Load environment variables
load_dotenv()

# Database connection
DB_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/heart_disease')
engine = create_engine(DB_URL)

def load_data():
    """Load the heart disease dataset."""
    print("Loading data...")
    df = pd.read_csv('data/Heart Prediction Quantum Dataset.csv')
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and duplicates."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna({
        'BloodPressure': df['BloodPressure'].median(),
        'Cholesterol': df['Cholesterol'].median(),
        'HeartRate': df['HeartRate'].median(),
        'QuantumPatternFeature': df['QuantumPatternFeature'].median()
    })
    
    return df

def engineer_features(df):
    """Create new features from existing data"""
    # Create age groups with finer granularity for higher risk ages
    age_bins = [0, 30, 40, 50, 55, 60, 65, 70, 100]
    age_labels = ['<30', '30-40', '40-50', '50-55', '55-60', '60-65', '65-70', '>70']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    # Create interaction features
    df['bp_age_interaction'] = df['BloodPressure'] * df['Age']
    df['chol_age_interaction'] = df['Cholesterol'] * df['Age']
    df['hr_bp_interaction'] = df['HeartRate'] * df['BloodPressure']
    
    # Create risk indicators with more granular thresholds
    df['high_bp_risk'] = (df['BloodPressure'] > 130).astype(int)
    df['high_chol_risk'] = (df['Cholesterol'] > 200).astype(int)
    df['high_hr_risk'] = (df['HeartRate'] > 100).astype(int)
    
    # Create combined risk score with more granular components
    df['risk_score'] = (
        df['high_bp_risk'] * 2 + 
        df['high_chol_risk'] * 2 + 
        df['high_hr_risk'] * 2 +
        (df['Age'] > 55).astype(int) * 3 +
        (df['Gender'] == 1).astype(int) * 2 +
        ((df['BloodPressure'] > 140).astype(int) +
        (df['Cholesterol'] > 240).astype(int) +
        (df['HeartRate'] > 120).astype(int)) * 2
    )
    
    # Create polynomial features for key numerical variables
    df['age_squared'] = df['Age'] ** 2
    df['bp_squared'] = df['BloodPressure'] ** 2
    df['chol_squared'] = df['Cholesterol'] ** 2
    
    # Create ratio features
    df['bp_chol_ratio'] = df['BloodPressure'] / df['Cholesterol']
    df['hr_bp_ratio'] = df['HeartRate'] / df['BloodPressure']
    
    # Create quantum pattern feature with more complexity
    df['quantum_pattern'] = (
        np.sin(df['Age'] * 0.1) * 
        np.cos(df['BloodPressure'] * 0.05) * 
        np.sin(df['Cholesterol'] * 0.02)
    )
    
    # Create derived features from quantum pattern
    df['quantum_pattern_squared'] = df['quantum_pattern'] ** 2
    df['quantum_pattern_cubed'] = df['quantum_pattern'] ** 3
    
    # Create categorical interaction features
    df['sex_age_group'] = df['Gender'].astype(str) + '_' + df['age_group'].astype(str)
    
    # Create risk level categories with handling for duplicate values
    try:
        df['risk_level'] = pd.qcut(df['risk_score'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    except ValueError:
        # If we can't create exactly 4 categories, use approximate quantile binning
        risk_thresholds = df['risk_score'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
        df['risk_level'] = pd.cut(df['risk_score'], 
                                 bins=risk_thresholds, 
                                 labels=['Low', 'Medium', 'High', 'Very High'],
                                 include_lowest=True)
    
    # Create health status indicators
    df['healthy_bp'] = (df['BloodPressure'] >= 90) & (df['BloodPressure'] <= 120)
    df['healthy_chol'] = (df['Cholesterol'] >= 120) & (df['Cholesterol'] <= 200)
    df['healthy_hr'] = (df['HeartRate'] >= 60) & (df['HeartRate'] <= 100)
    
    # Create composite health score
    df['health_score'] = (
        df['healthy_bp'].astype(int) + 
        df['healthy_chol'].astype(int) + 
        df['healthy_hr'].astype(int)
    )
    
    # Rename columns to match our schema
    column_mapping = {
        'Age': 'age',
        'Gender': 'sex',
        'BloodPressure': 'trestbps',
        'Cholesterol': 'chol',
        'HeartRate': 'thalach',
        'HeartDisease': 'target',
        'QuantumPatternFeature': 'quantum_pattern_raw'
    }
    
    # Drop any columns we don't want to save
    columns_to_keep = [
        'age', 'sex', 'trestbps', 'chol', 'thalach', 'target',
        'age_group', 'bp_age_interaction', 'chol_age_interaction',
        'hr_bp_interaction', 'high_bp_risk', 'high_chol_risk',
        'high_hr_risk', 'risk_score', 'age_squared', 'bp_squared',
        'chol_squared', 'bp_chol_ratio', 'hr_bp_ratio',
        'quantum_pattern', 'quantum_pattern_squared', 'quantum_pattern_cubed',
        'sex_age_group', 'risk_level', 'healthy_bp', 'healthy_chol',
        'healthy_hr', 'health_score'
    ]
    
    # Rename and select columns
    df = df.rename(columns=column_mapping)[columns_to_keep]
    
    return df

def save_to_database(df):
    """Save processed data to database"""
    # Create database connection
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql://localhost/heart_disease'))
    
    # Drop existing tables if they exist
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS processed_heart_data CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS model_predictions CASCADE"))
        conn.commit()
    
    # Create processed_heart_data table with all new features
    create_table_sql = """
    CREATE TABLE processed_heart_data (
        id SERIAL PRIMARY KEY,
        age INTEGER,
        sex INTEGER,
        trestbps INTEGER,
        chol INTEGER,
        thalach INTEGER,
        target INTEGER,
        age_group VARCHAR(10),
        bp_age_interaction FLOAT,
        chol_age_interaction FLOAT,
        hr_bp_interaction FLOAT,
        high_bp_risk INTEGER,
        high_chol_risk INTEGER,
        high_hr_risk INTEGER,
        risk_score INTEGER,
        age_squared FLOAT,
        bp_squared FLOAT,
        chol_squared FLOAT,
        bp_chol_ratio FLOAT,
        hr_bp_ratio FLOAT,
        quantum_pattern FLOAT,
        quantum_pattern_squared FLOAT,
        quantum_pattern_cubed FLOAT,
        sex_age_group VARCHAR(20),
        risk_level VARCHAR(10),
        healthy_bp BOOLEAN,
        healthy_chol BOOLEAN,
        healthy_hr BOOLEAN,
        health_score INTEGER
    )
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()
    
    # Insert data
    df.to_sql('processed_heart_data', engine, if_exists='append', index=False)

def main():
    """Main function to orchestrate the data preparation process."""
    print("Starting data preparation...")
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    # Clean data
    df = clean_data(df)
    print(f"After cleaning: {len(df)} records")
    
    # Engineer features
    df = engineer_features(df)
    print("Feature engineering completed")
    
    # Save to database
    save_to_database(df)
    print("Data saved to database")
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main() 