# ML-FORGE
Predictive Analysis For Structural Health Monitoring using python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv("sensor_data.csv")  # Assumes CSV with readings + damage label

# Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

# Encode labels if necessary
y = y.map({"undamaged": 0, "damaged": 1})

# Preprocessing: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model: Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(clf, "damage_classifier_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save processed dataset
processed_data = pd.DataFrame(X_scaled, columns=X.columns)
processed_data["label"] = y.values
processed_data.to_csv("processed_sensor_data.csv", index=False)

