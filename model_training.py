import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("data/train.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
train["IsAlone"] = (train["FamilySize"] == 1).astype(int)
train["Title"] = train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

train.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

for col in ["Sex", "Embarked", "Title"]:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])



from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = train.drop("Survived", axis=1)
y = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")