# Tableau Visualization Guide

## 1. Model Performance Dashboard

### Model Comparison Matrix
1. **Accuracy Comparison Bar Chart**
   ```sql
   -- Data Source: tableau_model_trends
   -- Dimensions: model_name
   -- Measures: accuracy
   -- Sort: accuracy (descending)
   ```
   - Chart Type: Horizontal Bar Chart
   - Color: Blue gradient
   - Add reference line for average accuracy
   - Include confidence interval bands

2. **Confusion Matrix Heat Map**
   ```sql
   -- Data Source: patient_risk_analysis
   -- Dimensions: 
   --   - prediction_accuracy (True/False)
   --   - risk_category
   -- Measures: COUNT
   ```
   - Chart Type: Heat Map
   - Color: Red-Green diverging
   - Add percentage labels
   - Include tooltips with detailed metrics

## 2. Patient Demographics Dashboard

### Age Distribution Analysis
1. **Age Group vs. Heart Disease Cases**
   ```sql
   -- Data Source: tableau_demographic_risk
   -- Dimensions: age_group
   -- Measures: 
   --   - total_patients
   --   - heart_disease_cases
   ```
   - Chart Type: Stacked Bar Chart
   - Color: Blue (healthy), Red (diseased)
   - Add percentage labels
   - Include trend line

2. **Age-Adjusted Risk Distribution**
   ```sql
   -- Data Source: tableau_risk_assessment
   -- Dimensions: age_group
   -- Measures: age_adjusted_risk
   ```
   - Chart Type: Box Plot
   - Color: Blue gradient
   - Add reference line at 1.0
   - Include outliers

### Gender Analysis
1. **Gender Distribution with Risk Categories**
   ```sql
   -- Data Source: tableau_patient_profiles
   -- Dimensions: 
   --   - sex
   --   - risk_category
   -- Measures: COUNT
   ```
   - Chart Type: Stacked Bar Chart
   - Color: Red gradient for risk levels
   - Add percentage labels
   - Include gender ratio calculation

## 3. Risk Analysis Dashboard

### Risk Score Distribution
1. **Risk Score Histogram**
   ```sql
   -- Data Source: tableau_patient_profiles
   -- Dimensions: risk_score (binned)
   -- Measures: COUNT
   ```
   - Chart Type: Histogram
   - Color: Red gradient
   - Add reference lines for risk thresholds
   - Include cumulative percentage

2. **Risk Trend Over Time**
   ```sql
   -- Data Source: tableau_model_trends
   -- Dimensions: prediction_date
   -- Measures: avg_risk_score
   ```
   - Chart Type: Line Chart
   - Color: Red
   - Add moving average
   - Include confidence bands

### Feature Importance
1. **Top Contributing Factors**
   ```sql
   -- Data Source: tableau_feature_importance
   -- Dimensions: feature_name
   -- Measures: importance
   -- Filter: rank <= 10
   ```
   - Chart Type: Horizontal Bar Chart
   - Color: Blue gradient
   - Add reference line for average importance
   - Include feature descriptions in tooltips

2. **Feature Correlation Matrix**
   ```sql
   -- Data Source: feature_correlations
   -- Dimensions: feature1, feature2
   -- Measures: correlation
   ```
   - Chart Type: Heat Map
   - Color: Red-Blue diverging
   - Add correlation values
   - Include significance indicators

## 4. Interactive Prediction Dashboard

### Patient Risk Assessment
1. **Risk Score Gauge**
   ```sql
   -- Data Source: tableau_risk_assessment
   -- Dimensions: risk_category
   -- Measures: risk_score
   ```
   - Chart Type: Gauge
   - Color: Red-Yellow-Green gradient
   - Add reference bands
   - Include confidence interval

2. **Risk Factors Analysis**
   ```sql
   -- Data Source: tableau_risk_assessment
   -- Dimensions: risk_factors
   -- Measures: COUNT
   ```
   - Chart Type: Treemap
   - Color: Red gradient
   - Size by count
   - Include percentage labels

### Historical Predictions
1. **Prediction Accuracy Trend**
   ```sql
   -- Data Source: tableau_model_trends
   -- Dimensions: prediction_date
   -- Measures: accuracy
   ```
   - Chart Type: Line Chart
   - Color: Green
   - Add moving average
   - Include model comparison

2. **Model Agreement Analysis**
   ```sql
   -- Data Source: model_comparison
   -- Dimensions: prediction_agreement
   -- Measures: COUNT
   ```
   - Chart Type: Pie Chart
   - Color: Green (Agreement), Red (Disagreement)
   - Add percentage labels
   - Include detailed breakdown

## Common Elements Across All Dashboards

### Filters
- Date Range
- Model Selection
- Risk Category
- Age Group
- Gender

### Parameters
```sql
-- Risk Thresholds
[Risk Threshold Low] = 0.2
[Risk Threshold Medium] = 0.4
[Risk Threshold High] = 0.6

-- Confidence Levels
[Confidence Threshold] = 0.8
```

### Calculations
```sql
-- Risk Score Normalization
[Risk Score Normalized] = 
    (risk_score - MIN(risk_score)) / (MAX(risk_score) - MIN(risk_score))

-- Prediction Confidence
[Prediction Confidence] = 
    IF prediction = actual_outcome THEN 1 ELSE 0 END

-- Age-Adjusted Risk
[Age-Adjusted Risk] = 
    risk_score / {FIXED [age_group]: AVG(risk_score)}
```

### Color Schemes
- Primary: #1f77b4 (Blue)
- Secondary: #d62728 (Red)
- Accent: #2ca02c (Green)
- Background: #f7f7f7 (Light Gray)
- Text: #2c3e50 (Dark Blue)

### Tooltips
Include for all visualizations:
- Detailed metrics
- Context information
- Trend indicators
- Comparison values 