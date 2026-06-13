import pandas as pd
import joblib

# Load the trained model
print("Loading model...")
model = joblib.load("model.pkl")

print("Loading dataset...")
data = pd.read_csv("creditcard.csv.csv")

# Take the first transaction as a sample
sample = data.drop("Class", axis=1).iloc[[0]]

# Predict
prediction = model.predict(sample)

# Display result
if prediction[0] == 0:
    print("✅ Normal Transaction")
else:
    print("❌ Fraud Transaction")