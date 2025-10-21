import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from kaggle.api.kaggle_api_extended import KaggleApi
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_mlflow():
    """Configure MLflow tracking"""
    # MLflow tracking URI is set via environment variable MLFLOW_TRACKING_URI
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    mlflow.set_tracking_uri(tracking_uri)
    
    # Wait for MLflow to be ready (give it up to 60 seconds)
    print(f"Waiting for MLflow server at {tracking_uri}...")
    max_retries = 12
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            # Try to set experiment - this will fail if MLflow isn't ready
            mlflow.set_experiment("titanic-survival-prediction")
            print(f"✓ Successfully connected to MLflow!")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}/{max_retries}: MLflow not ready yet, waiting {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"✗ Failed to connect to MLflow after {max_retries} attempts")
                raise
    
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: titanic-survival-prediction")

def load_dataset():
    """Download and load the Titanic dataset"""
    print("Downloading Titanic dataset from Kaggle...")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    dataset_path = '/app/data'
    os.makedirs(dataset_path, exist_ok=True)
    
    print("Downloading dataset files...")
    api.dataset_download_files('yasserh/titanic-dataset', path=dataset_path, unzip=True)
    print(f"Path to dataset files: {dataset_path}")
    
    # Find the CSV file in the downloaded path
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded dataset")
    
    # Load the first CSV file
    data_path = os.path.join(dataset_path, csv_files[0])
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from: {data_path}")
    print(f"Dataset shape: {df.shape}")
    
    return df

def preprocess_data(df):
    """Preprocess the Titanic dataset"""
    # Select relevant columns
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    target = 'Survived'
    
    # Create a copy with selected columns
    df_clean = df[features + [target]].copy()
    
    print(f"\nOriginal dataset shape: {df_clean.shape}")
    print(f"Missing values:\n{df_clean.isnull().sum()}")
    
    # Handle missing values
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
    
    # Drop any remaining rows with missing values
    df_clean.dropna(inplace=True)
    
    print(f"\nDataset shape after cleaning: {df_clean.shape}")
    
    # Encode categorical variable (Sex)
    le = LabelEncoder()
    df_clean['Sex'] = le.fit_transform(df_clean['Sex'])
    
    return df_clean, le

def create_visualizations(feature_importance_df, confusion_mat, run_name):
    """Create and save visualizations for MLflow"""
    viz_dir = '/tmp/mlflow_viz'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title(f'Feature Importance - {run_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    feature_plot_path = f'{viz_dir}/feature_importance.png'
    plt.savefig(feature_plot_path)
    plt.close()
    
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {run_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    confusion_plot_path = f'{viz_dir}/confusion_matrix.png'
    plt.savefig(confusion_plot_path)
    plt.close()
    
    return feature_plot_path, confusion_plot_path

def train_and_log_model(X_train, X_test, y_train, y_test, model, model_name, params):
    """Train model and log everything to MLflow"""
    
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Train model
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            mlflow.log_metric("roc_auc", roc_auc)
            print(f"ROC AUC: {roc_auc:.4f}")
        
        print(f"\nMetrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nFeature Importance:")
            print(feature_importance)
            
            # Create and log visualizations
            feature_plot, confusion_plot = create_visualizations(
                feature_importance, conf_matrix, model_name
            )
            mlflow.log_artifact(feature_plot, "plots")
            mlflow.log_artifact(confusion_plot, "plots")
        else:
            # For models without feature importance, still log confusion matrix
            viz_dir = '/tmp/mlflow_viz'
            os.makedirs(viz_dir, exist_ok=True)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            confusion_plot = f'{viz_dir}/confusion_matrix.png'
            plt.savefig(confusion_plot)
            plt.close()
            mlflow.log_artifact(confusion_plot, "plots")
        
        # Log the model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"titanic_{model_name.lower().replace(' ', '_')}"
        )
        
        # Log classification report as text artifact
        report = classification_report(y_test, y_pred)
        report_path = '/tmp/classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path, "reports")
        
        # Tag the run
        mlflow.set_tag("dataset", "titanic")
        mlflow.set_tag("task", "binary_classification")
        
        print(f"\nRun ID: {mlflow.active_run().info.run_id}")
        print(f"Model logged successfully!")
        
        return accuracy, model

def run_experiment_1_baseline():
    """Experiment 1: Baseline Random Forest"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE RANDOM FOREST")
    print("="*80)
    
    df = load_dataset()
    df_clean, label_encoder = preprocess_data(df)
    
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }
    
    accuracy, trained_model = train_and_log_model(
        X_train, X_test, y_train, y_test, 
        model, "Random Forest - Baseline", params
    )
    
    return X_train, X_test, y_train, y_test

def run_experiment_2_tuned_rf(X_train, X_test, y_train, y_test):
    """Experiment 2: Tuned Random Forest"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: TUNED RANDOM FOREST")
    print("="*80)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    
    train_and_log_model(
        X_train, X_test, y_train, y_test, 
        model, "Random Forest - Tuned", params
    )

def run_experiment_3_logistic_regression(X_train, X_test, y_train, y_test):
    """Experiment 3: Logistic Regression"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: LOGISTIC REGRESSION")
    print("="*80)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    params = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42
    }
    
    train_and_log_model(
        X_train, X_test, y_train, y_test, 
        model, "Logistic Regression", params
    )

def run_experiment_4_svm(X_train, X_test, y_train, y_test):
    """Experiment 4: Support Vector Machine"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: SUPPORT VECTOR MACHINE")
    print("="*80)
    
    model = SVC(kernel='rbf', probability=True, random_state=42)
    params = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "random_state": 42
    }
    
    train_and_log_model(
        X_train, X_test, y_train, y_test, 
        model, "SVM", params
    )

def main():
    """Main training pipeline with MLflow experiments"""
    print("=" * 80)
    print("TITANIC SURVIVAL PREDICTION - MLFLOW DEMO")
    print("=" * 80)
    
    # Setup MLflow
    setup_mlflow()
    
    # Run multiple experiments
    X_train, X_test, y_train, y_test = run_experiment_1_baseline()
    run_experiment_2_tuned_rf(X_train, X_test, y_train, y_test)
    run_experiment_3_logistic_regression(X_train, X_test, y_train, y_test)
    run_experiment_4_svm(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print("\nView results in MLflow UI at: http://localhost:5000")
    print("\nKey MLflow features demonstrated:")
    print("  ✓ Experiment tracking with multiple runs")
    print("  ✓ Parameter logging")
    print("  ✓ Metrics logging (accuracy, precision, recall, F1, ROC-AUC)")
    print("  ✓ Model logging and registration")
    print("  ✓ Artifact logging (plots, reports)")
    print("  ✓ Run comparison capabilities")
    print("=" * 80)

if __name__ == "__main__":
    main()

