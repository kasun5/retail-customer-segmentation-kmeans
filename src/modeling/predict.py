import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODEL_PATH
from src.dataset import impute_missing_values, load_dataset, parse_dates
from src.features import encode_categorical_features, engineer_features, transform_features


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load a trained model artifact created by ``train_model``."""
    with Path(path).open("rb") as model_file:
        return pickle.load(model_file)


def predict_clusters(
    data: pd.DataFrame | str | Path,
    model_path: str | Path = MODEL_PATH,
) -> pd.DataFrame:
    """Assign cluster labels while preserving each input row and its original values."""
    if isinstance(data, pd.DataFrame):
        original_df = parse_dates(data)
    else:
        original_df = load_dataset(data)

    payload = load_model(model_path)
    encoded_df = encode_categorical_features(original_df)
    imputed_df, _ = impute_missing_values(encoded_df, imputer=payload["imputer"])
    engineered_df = engineer_features(imputed_df)
    scaled_features = transform_features(engineered_df, payload["preprocessor"])
    pca_features = payload["pca"].transform(scaled_features)
    labels = payload["clusterer"].predict(pca_features) + 1

    predictions = original_df.copy()
    predictions["cluster_label"] = labels
    return predictions
