# Import libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Load dataset
project1= "Depression Professional Dataset.csv"
data = pd.read_csv(project1)

# Handle missing values
for col in data.columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)  # Fill missing values with median

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str).str.lower())
    label_encoders[col] = le  # Store encoders for later use

# Normalize numerical features
feature_columns = data.drop(columns=["Depression"]).columns
scaler = MinMaxScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Split dataset
X = data.drop(columns=["Depression"])
y = data["Depression"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Logistic Regression model with optimized parameters
model = LogisticRegression(max_iter=1000, C=1.5, solver='lbfgs', class_weight='balanced')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save trained model and scaler
joblib.dump(model, 'depression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(accuracy, 'model_accuracy.pkl')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print("Model, scaler, and accuracy saved successfully!")