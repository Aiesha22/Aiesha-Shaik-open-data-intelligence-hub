import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/train.csv")

sns.countplot(x="Exited", data=df)
plt.title("Customer Churn Distribution")
plt.show()

sns.histplot(df["Age"], bins=30)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x="Exited", y="Balance", data=df)
plt.title("Balance vs Churn")
plt.show()