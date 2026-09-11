import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


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

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.drop("customerID", axis=1)


# ==============================
# 3. Separate Features and Target
# ==============================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ==============================
# 4. Same Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 5. Load Saved Pipeline
# ==============================

model_path = "models/churn_model.pkl"

model = joblib.load(model_path)

print("\nSaved pipeline loaded successfully!")


# ==============================
# 6. Make Predictions
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 7. Accuracy
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")


# ==============================
# 8. Classification Report
# ==============================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==============================
# 9. Confusion Matrix
# ==============================

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))