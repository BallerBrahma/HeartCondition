import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import os
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

DB_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/heart_disease')
engine = create_engine(DB_URL)

def load_data():
    """Load the heart disease dataset."""
    print("Loading data...")
    df = pd.read_csv('data/Heart Prediction Quantum Dataset.csv')
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and duplicates."""
    df = df.drop_duplicates()
    
    df = df.fillna({
        'BloodPressure': df['BloodPressure'].median(),
        'Cholesterol': df['Cholesterol'].median(),
        'HeartRate': df['HeartRate'].median(),
        'QuantumPatternFeature': df['QuantumPatternFeature'].median()
    })
    
    return df

def engineer_features(df):
    """Create new features from existing data"""
    age_bins = [0, 30, 40, 50, 55, 60, 65, 70, 100]
    age_labels = ['<30', '30-40', '40-50', '50-55', '55-60', '60-65', '65-70', '>70']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    df['bp_age_interaction'] = df['BloodPressure'] * df['Age']
    df['chol_age_interaction'] = df['Cholesterol'] * df['Age']
    df['hr_bp_interaction'] = df['HeartRate'] * df['BloodPressure']
    
    df['high_bp_risk'] = (df['BloodPressure'] > 130).astype(int)
    df['high_chol_risk'] = (df['Cholesterol'] > 200).astype(int)
    df['high_hr_risk'] = (df['HeartRate'] > 100).astype(int)
    
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
    
    df['age_squared'] = df['Age'] ** 2
    df['bp_squared'] = df['BloodPressure'] ** 2
    df['chol_squared'] = df['Cholesterol'] ** 2
    
    df['bp_chol_ratio'] = df['BloodPressure'] / df['Cholesterol']
    df['hr_bp_ratio'] = df['HeartRate'] / df['BloodPressure']
    
    df['quantum_pattern'] = (
        np.sin(df['Age'] * 0.1) * 
        np.cos(df['BloodPressure'] * 0.05) * 
        np.sin(df['Cholesterol'] * 0.02)
    )
    
    df['quantum_pattern_squared'] = df['quantum_pattern'] ** 2
    df['quantum_pattern_cubed'] = df['quantum_pattern'] ** 3
    
    df['sex_age_group'] = df['Gender'].astype(str) + '_' + df['age_group'].astype(str)
    
    try:
        df['risk_level'] = pd.qcut(df['risk_score'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    except ValueError:
        risk_thresholds = df['risk_score'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
        df['risk_level'] = pd.cut(df['risk_score'], 
                                 bins=risk_thresholds, 
                                 labels=['Low', 'Medium', 'High', 'Very High'],
                                 include_lowest=True)
    
    df['healthy_bp'] = (df['BloodPressure'] >= 90) & (df['BloodPressure'] <= 120)
    df['healthy_chol'] = (df['Cholesterol'] >= 120) & (df['Cholesterol'] <= 200)
    df['healthy_hr'] = (df['HeartRate'] >= 60) & (df['HeartRate'] <= 100)
    
    df['health_score'] = (
        df['healthy_bp'].astype(int) + 
        df['healthy_chol'].astype(int) + 
        df['healthy_hr'].astype(int)
    )
    
    column_mapping = {
        'Age': 'age',
        'Gender': 'sex',
        'BloodPressure': 'trestbps',
        'Cholesterol': 'chol',
        'HeartRate': 'thalach',
        'HeartDisease': 'target',
        'QuantumPatternFeature': 'quantum_pattern_raw'
    }
    
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
    
    df = df.rename(columns=column_mapping)[columns_to_keep]
    
    return df

def save_to_database(df):
    """Save processed data to database"""
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql://localhost/heart_disease'))
    
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS processed_heart_data CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS model_predictions CASCADE"))
        conn.commit()
    
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
    
    df.to_sql('processed_heart_data', engine, if_exists='append', index=False)

def main():
    """Main function to orchestrate the data preparation process."""
    print("Starting data preparation...")
    
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    df = clean_data(df)
    print(f"After cleaning: {len(df)} records")
    
    df = engineer_features(df)
    print("Feature engineering completed")
    
    save_to_database(df)
    print("Data saved to database")
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main() 
