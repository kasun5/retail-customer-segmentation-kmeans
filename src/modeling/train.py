import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.config import (
    K_CANDIDATES,
    MODEL_PATH,
    PCA_VARIANCE,
    PREDICTIONS_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.dataset import impute_missing_values, load_dataset, remove_outliers
from src.features import (
    encode_categorical_features,
    engineer_features,
    fit_transform_features,
)

logger = logging.getLogger(__name__)


def _fallback_elbow(k_values: list[int], inertia: list[float]) -> int:
    values = np.asarray(inertia, dtype=float)
    if len(values) < 3 or np.isclose(values[0], values[-1]):
        return k_values[0]

    x = np.linspace(0.0, 1.0, len(values))
    normalized_y = (values - values[-1]) / (values[0] - values[-1])
    distance_from_chord = (1.0 - x) - normalized_y
    return k_values[int(np.argmax(distance_from_chord))]


def select_optimal_k(
    pca_features: np.ndarray,
    candidates: tuple[int, ...] = K_CANDIDATES,
) -> dict[str, Any]:
    """Vote across PCA-space metrics, falling back to elbow when no majority wins."""
    if pca_features.ndim != 2 or len(pca_features) < 3:
        raise ValueError("At least 3 PCA-transformed rows are required to evaluate k.")

    k_values = [k for k in candidates if 2 <= k < len(pca_features)]
    if not k_values:
        raise ValueError("No valid candidate cluster counts are available.")

    inertias: list[float] = []
    silhouette_scores: list[float] = []
    davies_bouldin_scores: list[float] = []
    calinski_harabasz_scores: list[float] = []

    for k in k_values:
        candidate = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = candidate.fit_predict(pca_features)
        inertias.append(float(candidate.inertia_))
        silhouette_scores.append(float(silhouette_score(pca_features, labels)))
        davies_bouldin_scores.append(float(davies_bouldin_score(pca_features, labels)))
        calinski_harabasz_scores.append(float(calinski_harabasz_score(pca_features, labels)))

    detected_elbow = KneeLocator(
        k_values, inertias, curve="convex", direction="decreasing"
    ).knee
    elbow_k = int(detected_elbow) if detected_elbow is not None else _fallback_elbow(k_values, inertias)

    recommendations = {
        "elbow": elbow_k,
        "silhouette": k_values[int(np.argmax(silhouette_scores))],
        "davies_bouldin": k_values[int(np.argmin(davies_bouldin_scores))],
        "calinski_harabasz": k_values[int(np.argmax(calinski_harabasz_scores))],
    }
    vote_counts = Counter(recommendations.values())
    winner, votes = vote_counts.most_common(1)[0]
    selected_k = winner if (votes > len(recommendations) / 2 and winner > 2) else max(recommendations.values())

    
    if selected_k < 3:
        raise ValueError(
            f"Insufficient clustering: selected k={selected_k}. "
            "At least 3 clusters are required; return to modeling and adjust preprocessing or the candidate k range."
        )
        

    return {
        "k_values": k_values,
        "inertia": inertias,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
        "calinski_harabasz_scores": calinski_harabasz_scores,
        "recommendations": recommendations,
        "elbow_k": elbow_k,
        "selected_k": selected_k,
    }


def train_model(
    raw_data_path: str | Path = RAW_DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    predictions_path: str | Path = PREDICTIONS_PATH,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the notebook's MICE, feature, PCA, and KMeans workflow and save outputs."""
    raw_df = load_dataset(raw_data_path)
    encoded_df = encode_categorical_features(raw_df)
    imputed_df, imputer = impute_missing_values(encoded_df)
    engineered_df = engineer_features(imputed_df)
    training_df = remove_outliers(engineered_df)

    scaled_features, preprocessor = fit_transform_features(training_df)
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    pca_features = pca.fit_transform(scaled_features)

    selection = select_optimal_k(pca_features)
    clusterer = KMeans(
        n_clusters=selection["selected_k"],
        random_state=RANDOM_STATE,
        n_init="auto",
    )
    labels = clusterer.fit_predict(pca_features)

    metrics = {
        "k": int(selection["selected_k"]),
        "silhouette": float(silhouette_score(pca_features, labels)),
        "davies_bouldin": float(davies_bouldin_score(pca_features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(pca_features, labels)),
        "inertia": float(clusterer.inertia_),
    }
    payload: dict[str, Any] = {
        "imputer": imputer,
        "preprocessor": preprocessor,
        "pca": pca,
        "clusterer": clusterer,
        "feature_columns": list(scaled_features.columns),
        "metrics": metrics,
        "k_selection": selection,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(payload, model_file)

    labeled_df = training_df.copy()
    labeled_df["cluster_label"] = labels + 1
    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(predictions_path, index=False)

    logger.info("Selected k=%s using %s", metrics["k"], selection["recommendations"])
    logger.info("Saved model to %s", model_path)
    logger.info("Saved clustered data to %s", predictions_path)
    return payload, labeled_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
