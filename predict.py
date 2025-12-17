import pandas as pd
import joblib
from features import create_features

model = joblib.load("models/model.joblib")

new_data = pd.read_csv("data/sample_input.csv")
new_data = create_features(new_data)

predictions = model.predict(new_data)
new_data["Prediction"] = predictions

new_data.to_csv("data/predictions.csv", index=False)
print("Predictions saved to data/predictions.csv")
