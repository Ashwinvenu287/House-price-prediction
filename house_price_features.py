"""Reusable, domain-based feature engineering for the Ames Housing project."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add a small set of interpretable housing features to raw Ames records."""

    engineered_features = [
        "HouseAge",
        "RemodAge",
        "GarageAge",
        "TotalSF",
        "TotalBathrooms",
        "TotalPorchSF",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()
        year_sold = pd.to_numeric(frame["YrSold"], errors="coerce")

        frame["HouseAge"] = (year_sold - pd.to_numeric(
            frame["YearBuilt"], errors="coerce"
        )).clip(lower=0)
        frame["RemodAge"] = (year_sold - pd.to_numeric(
            frame["YearRemodAdd"], errors="coerce"
        )).clip(lower=0)

        # A missing garage year means no garage in this dataset; encode its age as zero.
        garage_year = pd.to_numeric(frame["GarageYrBlt"], errors="coerce")
        frame["GarageAge"] = (year_sold - garage_year).clip(lower=0).where(
            garage_year.notna(), 0
        )

        frame["TotalSF"] = (
            frame["TotalBsmtSF"].fillna(0)
            + frame["1stFlrSF"].fillna(0)
            + frame["2ndFlrSF"].fillna(0)
        )
        frame["TotalBathrooms"] = (
            frame["FullBath"].fillna(0)
            + 0.5 * frame["HalfBath"].fillna(0)
            + frame["BsmtFullBath"].fillna(0)
            + 0.5 * frame["BsmtHalfBath"].fillna(0)
        )
        frame["TotalPorchSF"] = (
            frame["WoodDeckSF"].fillna(0)
            + frame["OpenPorchSF"].fillna(0)
            + frame["EnclosedPorch"].fillna(0)
            + frame["3SsnPorch"].fillna(0)
            + frame["ScreenPorch"].fillna(0)
        )
        return frame
