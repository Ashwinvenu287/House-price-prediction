# House Price Prediction

A machine learning project that predicts residential home sale prices using the Ames Housing dataset. The notebook explores the data, handles missing values, compares regression models, engineers housing features, and tunes model settings using cross-validation.

## Project files

- `Dataset/house-price-prediction.ipynb` — analysis and modeling notebook
- `Dataset/train.csv` and `Dataset/test.csv` — project datasets
- `Dataset/data_description.txt` — descriptions of the dataset features
- `house_price_features.py` — reusable housing feature engineering code

The notebook creates a trained model file under `models/` when run; that file is not needed to follow the notebook.

## Run the notebook

1. Install Python, Jupyter Notebook, and the required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, and `joblib`.
2. Open the `Dataset` folder as the notebook’s working directory.
3. Open `house-price-prediction.ipynb` and run the cells from top to bottom.

The notebook expects the CSV files in the `Dataset` folder and imports `house_price_features.py` from the project root. A `requirements.txt` file with package versions will be added separately.

## Model results

The tuned XGBoost model had the lowest average RMSE in the notebook’s five-fold cross-validation comparison: **$28,706**. This is a validation result, not a Kaggle leaderboard score.

## Dataset

The data is from Kaggle’s [House Prices: Advanced Regression Techniques competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).
