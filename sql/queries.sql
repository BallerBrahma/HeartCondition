-- Model Performance Analysis
CREATE OR REPLACE VIEW model_performance AS
SELECT 
    model_name,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as positive_predictions,
    SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as negative_predictions,
    AVG(probability) as avg_probability,
    MIN(probability) as min_probability,
    MAX(probability) as max_probability
FROM model_predictions
GROUP BY model_name;

-- Feature Importance Analysis
CREATE OR REPLACE VIEW feature_importance AS
WITH model_metrics AS (
    SELECT 
        model_name,
        jsonb_array_elements_text(metrics::jsonb) as metric
    FROM model_predictions
    GROUP BY model_name, metrics
)
SELECT 
    model_name,
    metric->>'name' as feature_name,
    (metric->>'importance')::float as importance
FROM model_metrics
WHERE metric->>'importance' IS NOT NULL
ORDER BY model_name, importance DESC;

-- Patient Risk Analysis
CREATE OR REPLACE VIEW patient_risk_analysis AS
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
    mp.probability as risk_score
FROM processed_heart_data p
JOIN model_predictions mp ON p.id = mp.patient_id;

-- Risk Score Distribution
CREATE OR REPLACE VIEW risk_score_distribution AS
SELECT 
    model_name,
    CASE 
        WHEN probability < 0.2 THEN 'Low Risk'
        WHEN probability < 0.4 THEN 'Moderate Risk'
        WHEN probability < 0.6 THEN 'High Risk'
        ELSE 'Very High Risk'
    END as risk_category,
    COUNT(*) as patient_count
FROM model_predictions
GROUP BY model_name, risk_category
ORDER BY model_name, risk_category;

-- Model Comparison
CREATE OR REPLACE VIEW model_comparison AS
SELECT 
    mp1.patient_id,
    mp1.model_name as model1_name,
    mp2.model_name as model2_name,
    mp1.prediction as model1_prediction,
    mp2.prediction as model2_prediction,
    mp1.probability as model1_probability,
    mp2.probability as model2_probability,
    CASE 
        WHEN mp1.prediction = mp2.prediction THEN 'Agreement'
        ELSE 'Disagreement'
    END as prediction_agreement
FROM model_predictions mp1
JOIN model_predictions mp2 ON mp1.patient_id = mp2.patient_id
WHERE mp1.model_name < mp2.model_name;

-- Patient Demographics Analysis
CREATE OR REPLACE VIEW patient_demographics AS
SELECT 
    age_group,
    sex,
    COUNT(*) as total_patients,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) as heart_disease_cases,
    AVG(CASE WHEN target = 1 THEN 1 ELSE 0 END) as heart_disease_rate
FROM processed_heart_data
GROUP BY age_group, sex
ORDER BY age_group, sex;

-- Feature Correlation Analysis
CREATE OR REPLACE VIEW feature_correlations AS
SELECT 
    feature1,
    feature2,
    correlation,
    sample_size
FROM (
    SELECT 
        'age_group' as feature1,
        'target' as feature2,
        corr(CAST(age_group AS INTEGER), target) as correlation,
        COUNT(*) as sample_size
    FROM processed_heart_data
    UNION ALL
    SELECT 
        'trestbps_normalized' as feature1,
        'target' as feature2,
        corr(trestbps_normalized, target) as correlation,
        COUNT(*) as sample_size
    FROM processed_heart_data
    -- Add more feature correlations as needed
) correlations
WHERE correlation IS NOT NULL
ORDER BY ABS(correlation) DESC; 