import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# STEP 2: CONVERT TOTALCHARGES TO NUMERIC
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# ============================================================
# STEP 3: CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# STEP 4: SEPARATE FEATURES AND TARGET
# ============================================================

# Remove Churn because it is our target.
# Remove customerID because it is only an identifier.

X = df.drop(
    columns=["Churn", "customerID"]
)

y = df["Churn"]


# ============================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nTraining Features Shape:")
print(X_train.shape)

print("Testing Features Shape:")
print(X_test.shape)

print("Training Target Shape:")
print(y_train.shape)

print("Testing Target Shape:")
print(y_test.shape)


# ============================================================
# STEP 6: CHURN DISTRIBUTION
# ============================================================

print("\nTraining Churn Distribution:")
print(y_train.value_counts())

print("\nTesting Churn Distribution:")
print(y_test.value_counts())


# ============================================================
# STEP 7: IDENTIFY NUMERICAL FEATURES
# ============================================================

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# STEP 8: IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    column
    for column in X.columns
    if column not in numeric_features
]


# ============================================================
# STEP 9: DISPLAY FEATURES
# ============================================================

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

print(
    "\nNumber of Numerical Features:",
    len(numeric_features)
)

print(
    "Number of Categorical Features:",
    len(categorical_features)
)
