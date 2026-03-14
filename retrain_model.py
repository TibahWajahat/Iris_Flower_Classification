"""
retrain_model.py
────────────────
Run this ONCE to regenerate iris_model.pkl and label_encoder.pkl
using your installed scikit-learn version (avoids pickle version mismatch).

Usage:
    python retrain_model.py
"""

import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("📂  Loading dataset …")
df = pd.read_csv("iris_flower_classification_10000_rows.csv")

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET   = "species"

X = df[FEATURES].values
y_raw = df[TARGET].values

print("🔠  Encoding labels …")
le = LabelEncoder()
y  = le.fit_transform(y_raw)
print(f"    Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🌳  Training Decision Tree …")
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅  Test Accuracy : {acc * 100:.2f}%")
print("\n📊  Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("💾  Saving iris_model.pkl …")
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾  Saving label_encoder.pkl …")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n🎉  Done! Both files saved. Now run: streamlit run app.py")