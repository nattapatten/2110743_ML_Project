import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import graphviz

# Function to load and prepare the data
def load_and_prepare_data(filepath, target_column):
    data = pd.read_csv(filepath)
    
    # Data Cleaning
    # Fill missing values with the mean of each column
    data.fillna(data.mean(), inplace=True)
    
    # Explore Dataset
    print(data.head())
    print(data.describe())
    print(data.info())
    
    # Split the data into features (X) and target (y)
    X = data.drop(target_column, axis=1)  # All columns except 'Outcome'
    y = data[target_column]  # Only the 'Outcome' column
    
    return X, y

# Function to split and scale the data
def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", roc_auc)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'Decision Tree (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# Function to perform hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']  # Removed 'auto'
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best parameters:", best_params)
    print("Best ROC-AUC score (cross-validation):", best_score)
    return grid_search.best_estimator_

# Function to visualize the tree using Graphviz
def visualize_tree(tree, feature_names):
    dot_data = export_graphviz(tree, out_file=None, 
                               feature_names=feature_names,  
                               class_names=['0', '1'],  
                               filled=True, rounded=True,  
                               special_characters=True)  
    graph = graphviz.Source(dot_data)  
    return graph

# Main code
if __name__ == "__main__":
    # Load and prepare the data
    X, y = load_and_prepare_data(r"..\\Datasets\\diabetes.csv", 'Outcome')
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)
    
    # Step 1: Initial Training and Evaluation
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    print("Initial Model Evaluation")
    evaluate_model(dt, X_test, y_test)
    
    # Step 2: Hyperparameter Tuning
    best_dt = hyperparameter_tuning(X_train, y_train)
    
    # Step 3: Re-evaluation with Optimized Hyperparameters
    print("Optimized Model Evaluation")
    evaluate_model(best_dt, X_test, y_test)
    
    # Visualize the decision tree using Graphviz
    graph = visualize_tree(dt, X.columns)
    graph.render("decision_tree")  # Save the visualization to a file
    graph.view()  # Display the visualization

    # Visualize the optimized decision tree using Graphviz
    graph_best = visualize_tree(best_dt, X.columns)
    graph_best.render("optimized_decision_tree")  # Save the visualization to a file
    graph_best.view()  # Display the visualization
