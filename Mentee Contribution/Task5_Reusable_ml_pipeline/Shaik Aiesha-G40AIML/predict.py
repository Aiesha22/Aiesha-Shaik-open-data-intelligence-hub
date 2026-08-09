import pandas as pd
import joblib


# ==============================
# 1. Load Saved Pipeline
# ==============================

model = joblib.load("models/churn_model.pkl")

print("Saved pipeline loaded successfully!")


# ==============================
# 2. Create New Customer
# ==============================

new_customer = pd.DataFrame({
    "gender": ["Female"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [5],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["DSL"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["No"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["No"],
    "StreamingMovies": ["No"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [70.5],
    "TotalCharges": [352.5]
})


# ==============================
# 3. Make Prediction
# ==============================

prediction = model.predict(new_customer)

probability = model.predict_proba(new_customer)


# ==============================
# 4. Display Result
# ==============================

print("\n==============================")
print("CUSTOMER CHURN PREDICTION")
print("==============================")

print("\nPrediction:", prediction[0])

print(
    f"Churn Probability: {probability[0][1]:.2%}"
)


# ==============================
# 5. Final Decision
# ==============================

if prediction[0] == "Yes":
    print("\n⚠️ Customer is likely to CHURN.")
else:
    print("\n✅ Customer is likely to STAY.")