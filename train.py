import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("data/train.csv")

# Basic preprocessing
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

# Encode gender
train["Sex"] = train["Sex"].map({
    "male": 0,
    "female": 1
})

# Encode embarked
train["Embarked"] = train["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

# Features
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = train[features]
y = train["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

test = pd.read_csv("data/test.csv")

test["Age"] = test["Age"].fillna(
    test["Age"].median()
)

test["Fare"] = test["Fare"].fillna(
    test["Fare"].median()
)

test["Sex"] = test["Sex"].map({
    "male": 0,
    "female": 1
})

test["Embarked"] = test["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

X_final = test[features]

predictions = model.predict(X_final)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)