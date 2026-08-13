from flask import Flask
import joblib

app = Flask(__name__)

model = joblib.load("models/churn_model.pkl")

@app.route("/")
def home():
    return "Customer Churn Forecasting Using AI"

if __name__ == "__main__":
    app.run(debug=True)