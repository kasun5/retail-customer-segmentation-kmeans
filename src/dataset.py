from pathlib import Path

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src.config import AGE_MAX, INCOME_MAX, MICE_RANDOM_STATE, RAW_DATA_PATH


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the customer enrollment date while preserving other columns."""
    result = df.copy()
    if "Dt_Customer" in result.columns:
        result["Dt_Customer"] = pd.to_datetime(
            result["Dt_Customer"], format="%d-%m-%Y", errors="coerce"
        )
    return result


def load_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw customer CSV and parse its date column."""
    return parse_dates(pd.read_csv(path))


def impute_missing_values(
    df: pd.DataFrame,
    imputer: IterativeImputer | None = None,
) -> tuple[pd.DataFrame, IterativeImputer]:
    """Fit or apply MICE to numeric columns, excluding identifiers and labels."""
    result = df.copy()

    if imputer is None:
        columns = [
            column
            for column in result.select_dtypes(include="number").columns
            if column not in {"ID", "cluster_label"}
        ]
        if not columns:
            raise ValueError("No numeric feature columns are available for imputation.")
        imputer = IterativeImputer(random_state=MICE_RANDOM_STATE)
        imputed = imputer.fit_transform(result[columns])
    else:
        columns = list(imputer.feature_names_in_)
        missing = [column for column in columns if column not in result.columns]
        if missing:
            raise ValueError(f"Missing columns required by the fitted imputer: {missing}")
        imputed = imputer.transform(result[columns])

    result.loc[:, columns] = imputed
    return result, imputer


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the age and income cutoffs used in the preprocessing notebook."""
    required = {"Age", "Income"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns required for outlier filtering: {missing}")

    return df.loc[(df["Age"] < AGE_MAX) & (df["Income"] < INCOME_MAX)].copy()
