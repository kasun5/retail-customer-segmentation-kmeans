import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import (
    EDUCATION_MAP,
    EDUCATION_ORDER,
    LOG_FEATURES,
    MARITAL_MAP,
    MARITAL_ORDER,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    REFERENCE_YEAR,
)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map and ordinal-encode education and marital status as in the notebook."""
    result = df.copy()
    categories = {
        "Education": (EDUCATION_MAP, EDUCATION_ORDER),
        "Marital_Status": (MARITAL_MAP, MARITAL_ORDER),
    }

    for column, (mapping, order) in categories.items():
        if column not in result.columns:
            raise ValueError(f"Required categorical column is missing: {column}")
        result[column] = result[column].replace(mapping)
        unknown = result[column].notna() & ~result[column].isin(order)
        if unknown.any():
            values = sorted(result.loc[unknown, column].astype(str).unique())
            raise ValueError(f"Unexpected {column} values: {values}")
        codes = pd.Categorical(result[column], categories=order, ordered=True).codes
        result[column] = np.where(codes < 0, np.nan, codes + 1)

    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer age, household, spending, and purchase features."""
    result = df.copy()
    required = {
        "Year_Birth", "Kidhome", "Teenhome", "Marital_Status", "Dt_Customer",
        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds", "NumDealsPurchases",
        "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4",
        "AcceptedCmp5",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"Missing columns required for feature engineering: {missing}")

    result["amountSpent"] = result[
        [
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
        ]
    ].sum(axis=1)
    result["numPurchases"] = result[
        [
            "NumDealsPurchases",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
        ]
    ].sum(axis=1)
    result["Age"] = REFERENCE_YEAR - result["Year_Birth"]
    result["children"] = result["Kidhome"] + result["Teenhome"]
    result["familySize"] = result["children"] + result["Marital_Status"]
    result["tenure_ytd"] = REFERENCE_YEAR - result["Dt_Customer"].dt.year

    campaign_columns = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
    ]
    result["AcceptedCmp"] = result[campaign_columns].sum(axis=1)
    result["is_AcceptedCmp"] = (result["AcceptedCmp"] > 0).astype(int)
    result["is_parent"] = (result["children"] > 0).astype(int)
    return result


def build_preprocessor() -> ColumnTransformer:
    """Build the numeric and log-scaling transformer used by the notebook."""
    numeric_pipeline = Pipeline([("standardize", StandardScaler())])
    log_pipeline = Pipeline(
        [
            (
                "log_transform",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
            ("standardize", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("log", log_pipeline, list(LOG_FEATURES)),
        ],
        remainder="drop",
    ).set_output(transform="pandas")


def _validate_model_features(df: pd.DataFrame) -> None:
    missing = [column for column in MODEL_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")


def fit_transform_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Fit feature scaling on a training frame and return transformed features."""
    _validate_model_features(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)
    return transformed, preprocessor


def transform_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    """Transform a frame with a previously fitted feature preprocessor."""
    _validate_model_features(df)
    return preprocessor.transform(df)
