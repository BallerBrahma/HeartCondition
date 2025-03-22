# Heart Disease Prediction Project

This project implements a comprehensive heart disease prediction system using SQL, Python, and Tableau. The system processes medical data, trains machine learning models, and provides interactive visualizations for analysis.

## Project Structure

```
.
├── data/                      # Data files
│   └── Heart Prediction Quantum Dataset.csv
├── sql/                      # SQL scripts
│   ├── setup.sql            # Database setup
│   └── queries.sql          # Analysis queries
├── src/                      # Python source code
│   ├── data_preparation.py  # Data cleaning and preprocessing
│   └── model_training.py    # ML model training
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up PostgreSQL database:
   - Install PostgreSQL if not already installed
   - Create a new database named 'heart_disease'
   - Run the SQL setup script:
     ```bash
     psql -d heart_disease -f sql/setup.sql
     ```

3. Run the data preparation script:
   ```bash
   python src/data_preparation.py
   ```

## Project Phases

1. **Data Preparation**
   - Data cleaning and preprocessing
   - Feature engineering
   - SQL database setup and data loading

2. **Predictive Modeling**
   - Model training and evaluation
   - Prediction generation
   - Results storage in SQL

3. **Visualization & Analysis**
   - Tableau dashboard creation
   - Interactive visualizations
   - Model performance metrics

## Technologies Used

- Python 3.x
- PostgreSQL
- Pandas
- Scikit-learn
- Tableau
- SQLAlchemy

## License

MIT License 