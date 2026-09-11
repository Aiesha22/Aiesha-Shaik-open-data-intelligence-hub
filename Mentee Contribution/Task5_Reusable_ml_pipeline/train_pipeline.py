import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# ==============================
# 1. Load Dataset
# ==============================

data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==============================
# 2. Data Cleaning
# ==============================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Drop customer ID
df = df.drop("customerID", axis=1)


# ==============================
# 3. Separate Features and Target
# ==============================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ==============================
# 4. Identify Feature Types
# ==============================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


print("\nNumerical features:")
print(list(numerical_features))

print("\nCategorical features:")
print(list(categorical_features))


# ==============================
# 5. Numerical Pipeline
# ==============================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ==============================
# 6. Categorical Pipeline
# ==============================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ==============================
# 7. Combine Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ==============================
# 8. Create Complete ML Pipeline
# ==============================

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# ==============================
# 9. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==============================
# 10. Train Pipeline
# ==============================

print("\nTraining model...")

model_pipeline.fit(X_train, y_train)

print("Model training completed!")


# ==============================
# 11. Create Models Folder
# ==============================

os.makedirs("models", exist_ok=True)


# ==============================
# 12. Save Complete Pipeline
# ==============================

model_path = "models/churn_model.pkl"

joblib.dump(model_pipeline, model_path)

print("\nComplete pipeline saved successfully!")
print("Model location:", model_path)