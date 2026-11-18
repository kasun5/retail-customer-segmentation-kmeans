"""
Customer Segmentation – Main Execution Script
==============================================
Runs MICE preprocessing, feature engineering, PCA, majority-vote optimal-k
selection, KMeans training, model persistence, and prediction export.
"""

import logging

from src.config import MODEL_PATH, PREDICTIONS_PATH
from src.modeling.train import train_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Running customer segmentation pipeline ===")
    payload, labeled_df = train_model()

    metrics = payload["metrics"]
    recommendations = payload["k_selection"]["recommendations"]
    sizes = labeled_df["cluster_label"].value_counts().sort_index()

    print("\n" + "=" * 60)
    print(f"  Customer Segmentation Results  (k={metrics['k']})")
    print("=" * 60)
    print("  Metric recommendations:")
    for method, k in recommendations.items():
        print(f"    {method:<20}: k={k}")
    print(f"  Silhouette Score     : {metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin Score : {metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz    : {metrics['calinski_harabasz']:.2f}")

    print("\nCluster Sizes:")
    for cluster, size in sizes.items():
        percentage = 100 * size / len(labeled_df)
        print(f"  Cluster {cluster}: {size:>5} customers ({percentage:.1f}%)")

    print("=" * 60)
    print(f"\nModel  → {MODEL_PATH}")
    print(f"Data   → {PREDICTIONS_PATH}")

    print("\nPipeline complete. You may now run the prediction script to assign cluster labels to new data.")

if __name__ == "__main__":
    main()
