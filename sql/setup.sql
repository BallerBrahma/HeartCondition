-- Create the raw data table
CREATE TABLE IF NOT EXISTS raw_heart_data (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    sex INTEGER,
    cp INTEGER,
    trestbps INTEGER,
    chol INTEGER,
    fbs INTEGER,
    restecg INTEGER,
    thalach INTEGER,
    exang INTEGER,
    oldpeak FLOAT,
    slope INTEGER,
    ca INTEGER,
    thal INTEGER,
    target INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the processed data table
CREATE TABLE IF NOT EXISTS processed_heart_data (
    id SERIAL PRIMARY KEY,
    age_group VARCHAR(20),
    sex VARCHAR(10),
    cp_type VARCHAR(50),
    trestbps_normalized FLOAT,
    chol_normalized FLOAT,
    fbs VARCHAR(5),
    restecg_type VARCHAR(50),
    thalach_normalized FLOAT,
    exang VARCHAR(5),
    oldpeak_normalized FLOAT,
    slope_type VARCHAR(50),
    ca_count INTEGER,
    thal_type VARCHAR(50),
    target INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the model predictions table
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES processed_heart_data(id),
    prediction INTEGER,
    probability FLOAT,
    model_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create a view for model training data
CREATE OR REPLACE VIEW model_training_data AS
SELECT 
    age_group,
    sex,
    cp_type,
    trestbps_normalized,
    chol_normalized,
    fbs,
    restecg_type,
    thalach_normalized,
    exang,
    oldpeak_normalized,
    slope_type,
    ca_count,
    thal_type,
    target
FROM processed_heart_data;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_raw_heart_data_target ON raw_heart_data(target);
CREATE INDEX IF NOT EXISTS idx_processed_heart_data_target ON processed_heart_data(target);
CREATE INDEX IF NOT EXISTS idx_model_predictions_patient_id ON model_predictions(patient_id); 