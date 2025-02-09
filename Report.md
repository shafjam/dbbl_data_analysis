# Bank Service Waiting Time Analysis

## Project Overview
This project analyzes waiting times for various banking services to optimize customer service and resource allocation. The analysis uses machine learning to predict waiting times and identify key factors affecting service efficiency.

## Key Findings

### Data Overview
- Dataset contains 266,832 records after cleaning
- 30 unique banking services
- Average waiting time: 16.68 minutes
- Median waiting time: 10.45 minutes

### Service Distribution
Top 5 services by volume:
1. Deposit (99,755 transactions)
2. Agent (51,742 transactions)
3. Withdrawal (41,441 transactions)
4. Bill (19,787 transactions)
5. Debit Card (16,036 transactions)

### Model Performance
Best performing models by RMSE:
1. XGBoost: 3.97 minutes (RMSE), 95.49% (R²)
2. Voting Ensemble: 4.02 minutes (RMSE), 95.37% (R²)
3. CatBoost: 4.11 minutes (RMSE), 95.15% (R²)
4. Random Forest: 4.22 minutes (RMSE), 94.89% (R²)

### Key Predictive Features
Top 5 most influential features:
1. 3-period rolling mean wait time (30.14%)
2. 5-period rolling mean wait time (18.63%)
3. 10-period rolling mean wait time (15.12%)
4. 3-hour rolling mean (9.80%)
5. 15-period rolling mean wait time (8.68%)

## Technical Implementation

### Data Preprocessing
1. Timestamp conversion and validation
2. Waiting time calculation and outlier removal using IQR method
3. Feature engineering:
   - Temporal features (hour, day, weekend flags)
   - Service load metrics
   - Rolling statistics
   - Service complexity scores

### Feature Engineering
1. Time-based features:
   - Hour of day
   - Day of week
   - Rush hour flag
   - Weekend indicator
   
2. Service load features:
   - Hourly service counts
   - Service-specific metrics
   - Rolling statistics (3, 5, 10, 15 period windows)

3. Service complexity metrics:
   - Average wait times
   - Standard deviations
   - Load-adjusted complexity scores

### Model Development
1. Model Architecture:
   - CatBoost Regressor
   - Random Forest Regressor
   - XGBoost Regressor
   - Voting Ensemble (weighted combination)

2. Training Approach:
   - 80-20 train-test split
   - Time series cross-validation (5 folds)
   - Feature scaling using RobustScaler

### Explainable AI (XAI) Implementation
1. Feature importance analysis using Random Forest
2. Error distribution analysis
3. Performance metrics across different service types

## Future Improvements

1. Model Enhancements:
   - Implement neural network architectures
   - Add more sophisticated ensemble methods
   - Explore time series-specific models

2. Feature Engineering:
   - Add weather data
   - Include seasonal patterns
   - Incorporate customer segmentation

3. Operational Improvements:
   - Real-time prediction pipeline
   - API development for service integration
   - Automated retraining pipeline

## Usage Instructions

### Prerequisites
```python
pandas
numpy
scikit-learn
catboost
xgboost
seaborn
matplotlib
shap
```

### Running the Analysis
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Update the data path in `main.py`
4. Run the analysis: `python main.py`

### Code Structure
```
├── XAI_dbbl_analysis.ipynb  # Current Ipynb instance 
├── main.py                  # Main execution script
├── data_processing.py       # Data preprocessing functions
├── feature_engineering.py   # Feature creation and selection
├── model_training.py        # Model development and training
├── evaluation.py            # Model evaluation and XAI
└── utils.py                 # Helper functions
```