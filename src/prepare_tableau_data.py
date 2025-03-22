import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DB_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/heart_disease')
engine = create_engine(DB_URL)

def create_tableau_views():
    """Create additional views optimized for Tableau visualization."""
    
    with engine.connect() as connection:
        connection.execute(text("""
        CREATE OR REPLACE VIEW tableau_model_trends AS
        SELECT 
            DATE_TRUNC('day', mp.created_at) as prediction_date,
            mp.model_name,
            COUNT(*) as total_predictions,
            AVG(CASE WHEN mp.prediction = p.target THEN 1 ELSE 0 END) as accuracy,
            AVG(mp.probability) as avg_risk_score
        FROM model_predictions mp
        JOIN processed_heart_data p ON mp.patient_id = p.id
        GROUP BY DATE_TRUNC('day', mp.created_at), mp.model_name
        ORDER BY prediction_date, mp.model_name;
        """))
        
        connection.execute(text("""
        CREATE OR REPLACE VIEW tableau_patient_profiles AS
        SELECT 
            p.id,
            p.age_group,
            p.sex,
            p.cp_type,
            p.trestbps_normalized,
            p.chol_normalized,
            p.fbs,
            p.restecg_type,
            p.thalach_normalized,
            p.exang,
            p.oldpeak_normalized,
            p.slope_type,
            p.ca_count,
            p.thal_type,
            p.target as actual_outcome,
            mp.model_name,
            mp.prediction as predicted_outcome,
            mp.probability as risk_score,
            CASE 
                WHEN mp.probability < 0.2 THEN 'Low Risk'
                WHEN mp.probability < 0.4 THEN 'Moderate Risk'
                WHEN mp.probability < 0.6 THEN 'High Risk'
                ELSE 'Very High Risk'
            END as risk_category,
            CASE 
                WHEN mp.prediction = p.target THEN 'Correct'
                ELSE 'Incorrect'
            END as prediction_accuracy
        FROM processed_heart_data p
        JOIN model_predictions mp ON p.id = mp.patient_id;
        """))
        
        connection.execute(text("""
        CREATE OR REPLACE VIEW tableau_demographic_risk AS
        SELECT 
            p.age_group,
            p.sex,
            COUNT(*) as total_patients,
            SUM(CASE WHEN p.target = 1 THEN 1 ELSE 0 END) as heart_disease_cases,
            AVG(CASE WHEN p.target = 1 THEN 1 ELSE 0 END) as heart_disease_rate,
            AVG(mp.probability) as avg_risk_score,
            MIN(mp.probability) as min_risk_score,
            MAX(mp.probability) as max_risk_score
        FROM processed_heart_data p
        JOIN model_predictions mp ON p.id = mp.patient_id
        GROUP BY p.age_group, p.sex
        ORDER BY p.age_group, p.sex;
        """))
        
        connection.execute(text("""
        CREATE OR REPLACE VIEW tableau_prediction_confidence AS
        SELECT 
            mp.model_name,
            CASE 
                WHEN mp.probability < 0.2 THEN 'Very Low'
                WHEN mp.probability < 0.4 THEN 'Low'
                WHEN mp.probability < 0.6 THEN 'Medium'
                WHEN mp.probability < 0.8 THEN 'High'
                ELSE 'Very High'
            END as confidence_level,
            COUNT(*) as prediction_count,
            AVG(CASE WHEN mp.prediction = p.target THEN 1 ELSE 0 END) as accuracy_rate
        FROM model_predictions mp
        JOIN processed_heart_data p ON mp.patient_id = p.id
        GROUP BY mp.model_name, confidence_level
        ORDER BY mp.model_name, confidence_level;
        """))
        
        connection.commit()

def create_calculated_fields():
    """Create calculated fields for Tableau."""
    
    with engine.connect() as connection:
        connection.execute(text("""
        CREATE OR REPLACE VIEW tableau_risk_assessment AS
        SELECT 
            p.*,
            mp.model_name,
            mp.prediction,
            mp.probability as risk_score,
            CASE 
                WHEN mp.probability < 0.2 THEN 'Low Risk'
                WHEN mp.probability < 0.4 THEN 'Moderate Risk'
                WHEN mp.probability < 0.6 THEN 'High Risk'
                ELSE 'Very High Risk'
            END as risk_category,
            CASE 
                WHEN mp.prediction = p.target THEN 'Correct'
                ELSE 'Incorrect'
            END as prediction_accuracy,
            -- Additional calculated fields
            CASE 
                WHEN p.age_group IN ('>60', '51-60') AND mp.probability > 0.6 THEN 'High Risk Age Group'
                WHEN p.exang = 'Yes' AND mp.probability > 0.6 THEN 'High Risk with Exercise Angina'
                ELSE 'Standard Risk'
            END as risk_factors,
            -- Normalized risk score by age group
            mp.probability / AVG(mp.probability) OVER (PARTITION BY p.age_group) as age_adjusted_risk
        FROM processed_heart_data p
        JOIN model_predictions mp ON p.id = mp.patient_id;
        """))
        
        connection.commit()

def main():
    """Main function to prepare data for Tableau."""
    print("Starting Tableau data preparation...")
    
    create_tableau_views()
    print("Created Tableau-specific views")
    
    create_calculated_fields()
    print("Created calculated fields")
    
    print("\nTableau data preparation completed successfully!")
    print("You can now connect to these views in Tableau Desktop")

if __name__ == "__main__":
    main() 
