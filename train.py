import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from features import create_features
from preprocess import build_preprocessor

df = pd.read_csv("raw_data.csv")

df = create_features(df)

X = df.drop("Target", axis=1)
y = df["Target"]

num_features = ["Age", "Salary", "Salary_to_Age"]
cat_features = ["Department", "Age_Bucket"]

preprocessor = build_preprocessor(num_features, cat_features)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

joblib.dump(model, "models/model.joblib")
print("Model saved successfully")
