# Heart Disease Prediction Dashboard

This directory contains the Tableau workbook and connection details for the Heart Disease Prediction dashboard.

## Database Connection Details

1. Open Tableau Desktop
2. Click "Connect to Data"
3. Select "PostgreSQL"
4. Enter the following connection details:
   - Server: localhost
   - Port: 5432
   - Database: heart_disease
   - Authentication: Username and Password

## Dashboard Layout

### 1. Model Performance Dashboard
- **Model Comparison Matrix**
  - Compare accuracy, precision, recall, and F1-score across models
  - Use bar charts and heat maps
  - Add confidence intervals

- **Confusion Matrix Visualization**
  - True Positives, False Positives, True Negatives, False Negatives
  - Color-coded matrix with percentages

### 2. Patient Demographics Dashboard
- **Age Distribution**
  - Age groups vs. heart disease cases
  - Stacked bar chart showing positive/negative cases

- **Gender Analysis**
  - Gender distribution of heart disease cases
  - Pie charts and bar charts

### 3. Risk Analysis Dashboard
- **Risk Score Distribution**
  - Distribution of risk scores across models
  - Histogram with risk categories
  - Trend analysis over time

- **Feature Importance**
  - Top contributing factors to heart disease
  - Bar chart showing feature importance scores
  - Interactive filters for different models

### 4. Interactive Prediction Dashboard
- **Patient Risk Assessment**
  - Input form for new patient data
  - Real-time risk prediction
  - Confidence score visualization

- **Historical Predictions**
  - Success rate of predictions
  - Time series analysis of prediction accuracy

## Data Sources

The dashboard uses the following database views:
1. `model_performance`
2. `patient_demographics`
3. `risk_score_distribution`
4. `feature_importance`
5. `patient_risk_analysis`
6. `model_comparison`
7. `feature_correlations`

## Setup Instructions

1. Ensure PostgreSQL is running and accessible
2. Verify database connection details in `.env` file
3. Open the Tableau workbook
4. Refresh all data sources
5. Verify all calculations and parameters

## Best Practices

1. Use consistent color schemes across all dashboards
2. Implement filters for interactive analysis
3. Include tooltips with detailed information
4. Use appropriate chart types for different data types
5. Maintain consistent formatting and styling

## Troubleshooting

If you encounter connection issues:
1. Verify PostgreSQL is running
2. Check database credentials
3. Ensure all required views exist
4. Verify network connectivity
5. Check Tableau logs for error messages 