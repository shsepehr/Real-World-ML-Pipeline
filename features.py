import pandas as pd

def create_features(df):
    df = df.copy()
    
    df["Salary_to_Age"] = df["Salary"] / df["Age"]
    df["Age_Bucket"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 100],
        labels=["Young", "Mid", "Senior"]
    )
    
    return df
