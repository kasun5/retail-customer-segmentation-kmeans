from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "customer_segmentation.csv"
INTERIM_DATA_PATH = DATA_DIR / "interim" / "customer_segmentation_interim.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "customer_segmentation_processed.csv"
PREDICTIONS_PATH = DATA_DIR / "predictions" / "customer_segmentation_predictions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "customer_segmentation_model.pkl"

REFERENCE_YEAR = 2021
MICE_RANDOM_STATE = 99
RANDOM_STATE = 42
PCA_VARIANCE = 0.85
K_CANDIDATES = tuple(range(2, 8))
AGE_MAX = 90
INCOME_MAX = 200_000

EDUCATION_MAP = {
    "2n Cycle": "Master",
    "Basic": "High School",
    "Graduation": "Bachelor",
}
EDUCATION_ORDER = ("High School", "Bachelor", "Master", "PhD")

MARITAL_MAP = {
    "Absurd": "Single",
    "Alone": "Single",
    "Divorced": "Single",
    "Together": "Married",
    "Widow": "Single",
    "YOLO": "Single",
}
MARITAL_ORDER = ("Single", "Married")

NUMERIC_FEATURES = (
    "Education",
    "Income",
    "Recency",
    "Kidhome",
    "Teenhome",
    "children",
    "familySize",
    "is_parent",
    "Marital_Status",
    "Age",
)
LOG_FEATURES = (
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "amountSpent",
    "numPurchases",
)
MODEL_FEATURES = NUMERIC_FEATURES + LOG_FEATURES
PROFILE_FEATURES = (
    "Income",
    "Age",
    "amountSpent",
    "numPurchases",
    "NumWebVisitsMonth",
    "Recency",
    "children",
    "familySize",
)
