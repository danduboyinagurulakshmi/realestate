# Real Estate Investment Advisor

This capstone project builds an end-to-end machine learning system that helps investors judge if a property is a strong investment and estimates its five-year resale value.

## Business Objective

- Classify whether a property is a `Good Investment`.
- Predict the expected property price after five years.
- Deliver a Streamlit dashboard with investment insights.
- Track experiments with MLflow.

## Project Flow

1. Dataset loading and cleaning.
2. Feature engineering and investment label creation.
3. Exploratory data analysis.
4. Training classification and regression models.
5. MLflow experiment tracking.
6. Streamlit-based user interaction and predictions.

## Files

- `india_housing_prices.csv`: property dataset
- `train_models.py`: trains the models and saves them
- `streamlit_app.py`: interactive investment dashboard
- `src/real_estate_pipeline.py`: preprocessing, feature engineering, modeling, and prediction utilities
- `models/`: generated trained artifacts (created by `python train_models.py`; not committed)
- `plots/`: generated charts and summary JSON

## Key ML components

- Classification target: `Good_Investment`
- Regression target: `Five_Year_Projected_Price`
- Models: `RandomForestClassifier` and `RandomForestRegressor`
- Feature engineering includes amenities count, transport score, parking/security indicators, and value-per-square-foot signals.

## Run locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run streamlit_app.py
```

Training artifacts and MLflow run data are generated locally and excluded from Git to keep the repository lightweight.

## Results to expect

- Classification accuracy above 0.75 on a held-out test set depending on data split.
- Regression RMSE measured against the projected 5-year property value.
- Interactive dashboard for new property evaluation.

## EDA themes covered

- Average price by city and state.
- Property size vs price trends.
- Price-per-square-foot by property type.
- Correlation of amenity and transport features with value.
- Inventory distribution by BHK, owner type, and availability status.
