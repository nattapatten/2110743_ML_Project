import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc 
import joblib
import os
from datetime import datetime



# Get the directory of the current script dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the base folder relative to the script directory
base_folder = os.path.join(script_dir, 'Trained_Model', 'Logistic_Regression')
# Load the dataset
data = pd.read_csv(r"..\\Datasets\\diabetes.csv")


# Get the current date and time
current_date = datetime.now().strftime('%Y-%m-%d')
current_time = datetime.now().strftime('%H%M')

# Construct the full dynamic folder path
dynamic_folder_path = os.path.join(base_folder, current_date, current_time)

# Create the folder structure if it doesn't exist
os.makedirs(dynamic_folder_path, exist_ok=True)

# Save the model and scaler in the dynamically created folder
model_path = os.path.join(dynamic_folder_path, 'best_logistic_model.pkl')
scaler_path = os.path.join(dynamic_folder_path, 'scaler.pkl')

# Display column names to verify the target column
print(data.columns)

# Define the target column
target_column = 'Outcome'  # Replace 'Outcome' if your target column has a different name


# Data cleaning
data.fillna(data.mean(), inplace=True)


# Explore Dataset
print(data.head())
print(data.describe())
print(data.info())




# Split the data into features (X) and target (y)
X = data.drop(target_column, axis=1)  # All columns except 'Outcome'
y = data[target_column]  # Only the 'Outcome' column

# Verify the split
print(X.head())
print(y.head())


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale/normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 1: Initial Training and Evaluation

# Initialize the Logistic Regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Evaluate the model
print("Initial Model Evaluation")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_prob)
print("Initial ROC-AUC:", roc_auc)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'Logistic Regression (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 2: Hyperparameter Tuning

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='roc_auc')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters:", best_params)
print("Best ROC-AUC score (cross-validation):", best_score)

# Step 3: Re-evaluation with Optimized Hyperparameters

# Train the final model with the best parameters
best_logreg = grid_search.best_estimator_
best_logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_best = best_logreg.predict(X_test)
y_prob_best = best_logreg.predict_proba(X_test)[:, 1]  # Probabilities for the positive class


# Save the model and scaler in the dynamically created folder
joblib.dump(best_logreg, model_path)
joblib.dump(scaler, scaler_path)

print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")

# Evaluate the model
print("Optimized Model Evaluation")
print(classification_report(y_test, y_pred_best))
cm_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Optimized Model)')
plt.show()

# ROC-AUC score
roc_auc_best = roc_auc_score(y_test, y_prob_best)
print("Optimized ROC-AUC:", roc_auc_best)

# ROC Curve
fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_best)
plt.plot(fpr_best, tpr_best, label=f'Logistic Regression (area = {roc_auc_best:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Optimized Model)')
plt.legend()
plt.show()
