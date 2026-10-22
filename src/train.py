import os
import time
import uuid

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities import Metric, RunTag
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
)
from kaggle.api.kaggle_api_extended import KaggleApi
import matplotlib.pyplot as plt
import seaborn as sns

EXPERIMENT_NAME = "titanic-survival-prediction"
REGISTERED_MODEL_NAME = "titanic-survival"
DATASET_URL = "https://www.kaggle.com/datasets/yasserh/titanic-dataset"

NUMERIC_FEATURES = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
CATEGORICAL_FEATURES = ['Sex']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'Survived'

# Tags every candidate run in this training session so we can find them later
SESSION_ID = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

EXPERIMENT_DESCRIPTION = """
# Titanic Survival Prediction

Predict whether a passenger survived the Titanic disaster.

* **Dataset:** [Kaggle - yasserh/titanic-dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
* **Features:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`
* **Primary metric:** ROC-AUC on a held-out 20% test split

Every candidate model is registered as a version of **`titanic-survival`**;
the best one gets the `@champion` alias, the runner-up `@challenger`.
"""


def banner(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


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
            experiment = mlflow.set_experiment(EXPERIMENT_NAME)
            print(f"✓ Successfully connected to MLflow!")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}/{max_retries}: MLflow not ready yet, waiting {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"✗ Failed to connect to MLflow after {max_retries} attempts")
                raise

    # Experiment description and tags show up at the top of the experiment page
    client = MlflowClient()
    client.set_experiment_tag(experiment.experiment_id, "mlflow.note.content", EXPERIMENT_DESCRIPTION)
    client.set_experiment_tag(experiment.experiment_id, "project", "mlflow-demo")
    client.set_experiment_tag(experiment.experiment_id, "task", "binary_classification")

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"Training session: {SESSION_ID}")


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


def split_data(df):
    """Select features and split into train/test.

    Missing values and encoding are handled inside the model pipeline, so the
    logged models accept raw passenger rows.
    """
    df_clean = df[FEATURES + [TARGET]].dropna(subset=[TARGET])
    print(f"\nDataset shape: {df_clean.shape}")
    print(f"Missing values:\n{df_clean.isnull().sum()}")

    X = df_clean[FEATURES]
    y = df_clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MLflow datasets: shown in the "Datasets used" column and on each run page
    train_dataset = mlflow.data.from_pandas(
        X_train.assign(**{TARGET: y_train}), source=DATASET_URL, name="titanic-train", targets=TARGET
    )
    test_dataset = mlflow.data.from_pandas(
        X_test.assign(**{TARGET: y_test}), source=DATASET_URL, name="titanic-test", targets=TARGET
    )

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "train_dataset": train_dataset, "test_dataset": test_dataset,
    }


def build_pipeline(classifier):
    """Preprocessing + classifier in a single sklearn Pipeline"""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def log_feature_importance(pipeline, run_name):
    """Log a feature importance (or coefficient) chart, if the model has one"""
    classifier = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values, label = classifier.feature_importances_, "Importance"
    elif hasattr(classifier, "coef_"):
        values, label = classifier.coef_[0], "Coefficient"
    else:
        return

    importance = pd.DataFrame({"feature": feature_names, "value": values})
    importance = importance.sort_values("value", key=abs, ascending=False)
    print(f"\nFeature {label.lower()}:")
    print(importance.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance, x="value", y="feature", ax=ax)
    ax.set_title(f"Feature {label} - {run_name}")
    ax.set_xlabel(label)
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)


def log_learning_curve(pipeline, data):
    """Log per-boosting-stage metrics, which MLflow draws as line charts"""
    preprocess = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["model"]
    X_train = preprocess.transform(data["X_train"])
    X_test = preprocess.transform(data["X_test"])

    run_id = mlflow.active_run().info.run_id
    timestamp = int(time.time() * 1000)
    metrics = []
    stages = zip(classifier.staged_predict_proba(X_train), classifier.staged_predict_proba(X_test))
    for step, (train_proba, test_proba) in enumerate(stages, start=1):
        metrics += [
            Metric("train_log_loss", log_loss(data["y_train"], train_proba), timestamp, step),
            Metric("test_log_loss", log_loss(data["y_test"], test_proba), timestamp, step),
            Metric("test_accuracy_by_stage", accuracy_score(data["y_test"], test_proba.argmax(axis=1)), timestamp, step),
        ]

    # log_batch accepts up to 1000 metrics per call
    client = MlflowClient()
    for i in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[i:i + 1000])


def log_and_evaluate_model(pipeline, data, run_name):
    """Log the model with signature, evaluate it, and mark the run as a candidate"""
    X_train, X_test, y_test = data["X_train"], data["X_test"], data["y_test"]

    # Signature + input example show up on the model's artifact page
    signature = infer_signature(X_train, pipeline.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        pipeline,
        "model",
        signature=signature,
        input_example=X_train.head(5),
    )

    # mlflow.evaluate computes standard metrics and logs ROC, PR, calibration,
    # lift and confusion matrix plots as artifacts
    result = mlflow.evaluate(
        model_info.model_uri,
        data=data["test_dataset"],
        model_type="classifier",
        evaluator_config={"log_model_explainability": False},
    )

    report = classification_report(y_test, pipeline.predict(X_test))
    mlflow.log_text(report, "reports/classification_report.txt")

    mlflow.set_tag("candidate", "true")
    mlflow.set_tag("training_session", SESSION_ID)
    mlflow.set_tag("dataset", "titanic")
    mlflow.set_tag("task", "binary_classification")

    metrics = result.metrics
    print(f"\nMetrics:")
    for name in ["accuracy_score", "precision_score", "recall_score", "f1_score", "roc_auc"]:
        print(f"{name}: {metrics[name]:.4f}")
    print(f"\nRun ID: {mlflow.active_run().info.run_id}")
    print(f"Model logged successfully!")

    return metrics


def train_and_log_model(data, classifier, run_name, params, description, learning_curve=False):
    """Train model and log everything to MLflow"""
    with mlflow.start_run(run_name=run_name):
        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")

        # Run description shows at the top of the run page
        mlflow.set_tag("mlflow.note.content", description)
        mlflow.set_tag("model_type", type(classifier).__name__)
        mlflow.log_input(data["train_dataset"], context="training")

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(data["X_train"]))
        mlflow.log_param("test_size", len(data["X_test"]))

        # Train model
        print("Training model...")
        pipeline = build_pipeline(classifier)
        pipeline.fit(data["X_train"], data["y_train"])

        if learning_curve:
            log_learning_curve(pipeline, data)
        log_feature_importance(pipeline, run_name)

        return log_and_evaluate_model(pipeline, data, run_name)


def run_experiment_1_baseline(data):
    """Experiment 1: Baseline Random Forest"""
    banner("EXPERIMENT 1: BASELINE RANDOM FOREST")

    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }
    train_and_log_model(
        data, RandomForestClassifier(**params), "Random Forest - Baseline", params,
        "Random forest with **default hyperparameters**. This is the baseline every other run is compared against.",
    )


def run_experiment_2_tuned_rf(data):
    """Experiment 2: Tuned Random Forest"""
    banner("EXPERIMENT 2: TUNED RANDOM FOREST")

    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    train_and_log_model(
        data, RandomForestClassifier(**params), "Random Forest - Tuned", params,
        "Random forest with **hand-picked hyperparameters**: more trees, limited depth and "
        "larger leaves to reduce overfitting compared to the baseline.",
    )


def run_experiment_3_logistic_regression(data):
    """Experiment 3: Logistic Regression"""
    banner("EXPERIMENT 3: LOGISTIC REGRESSION")

    params = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42
    }
    train_and_log_model(
        data, LogisticRegression(**params), "Logistic Regression", params,
        "A simple, interpretable **linear model**. The feature chart shows coefficients: "
        "positive values increase the predicted odds of survival.",
    )


def run_experiment_4_svm(data):
    """Experiment 4: Support Vector Machine"""
    banner("EXPERIMENT 4: SUPPORT VECTOR MACHINE")

    params = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "random_state": 42
    }
    train_and_log_model(
        data, SVC(**params), "SVM", params,
        "RBF-kernel **support vector machine**. SVMs are sensitive to feature scale, "
        "so the pipeline standardizes numeric features first.",
    )


def run_experiment_5_gradient_boosting(data):
    """Experiment 5: Gradient Boosting with per-stage metrics"""
    banner("EXPERIMENT 5: GRADIENT BOOSTING")

    params = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "random_state": 42
    }
    train_and_log_model(
        data, GradientBoostingClassifier(**params), "Gradient Boosting", params,
        "Gradient boosted trees. Train/test log loss is logged **after every boosting stage**: "
        "open the *Model metrics* / *Charts* tab to see the learning curves.",
        learning_curve=True,
    )


def test_metrics(pipeline, data):
    """Same test-set metrics (and names) that mlflow.evaluate logs for a classifier"""
    y_test = data["y_test"]
    y_pred = pipeline.predict(data["X_test"])
    y_proba = pipeline.predict_proba(data["X_test"])[:, 1]
    return {
        "accuracy_score": accuracy_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "recall_score": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
    }


def log_child_test_metrics(search, param_grid, data):
    """Autolog only records CV scores on child runs; add test metrics so the runs table is comparable"""
    parent_run_id = mlflow.active_run().info.run_id
    children = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    client = MlflowClient()
    for _, child in children.iterrows():
        # Autolog stores params as strings; map them back to the grid values
        params = {
            name: next(v for v in values if str(v) == child[f"params.{name}"])
            for name, values in param_grid.items()
        }
        pipeline = clone(search.estimator).set_params(**params).fit(data["X_train"], data["y_train"])
        timestamp = int(time.time() * 1000)
        client.log_batch(
            child.run_id,
            metrics=[Metric(k, v, timestamp, 0) for k, v in test_metrics(pipeline, data).items()],
            tags=[
                RunTag("training_session", SESSION_ID),
                # Replace autolog's random run name with the parameter combination
                RunTag("mlflow.runName", "RF depth={model__max_depth} leaf={model__min_samples_leaf} "
                                         "trees={model__n_estimators}".format(**params)),
            ],
        )
    print(f"Logged test metrics to {len(children)} child runs")


def run_experiment_6_grid_search(data):
    """Experiment 6: Random Forest grid search with autologging"""
    banner("EXPERIMENT 6: RANDOM FOREST GRID SEARCH (AUTOLOG)")

    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [1, 3],
    }
    search = GridSearchCV(
        build_pipeline(RandomForestClassifier(random_state=42)),
        param_grid, cv=5, scoring="roc_auc", n_jobs=-1,
    )

    with mlflow.start_run(run_name="Random Forest - Grid Search"):
        mlflow.set_tag("mlflow.note.content",
            "Hyperparameter search with `GridSearchCV` and **`mlflow.sklearn.autolog()`**. "
            "Autolog creates one **child run per parameter combination** (expand this run in the "
            "runs table). Select the child runs and click *Compare* to see the parallel coordinates plot."
        )
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Autolog records params, CV results and one nested run per combination.
        # Models are logged by log_and_evaluate_model so every candidate looks the same.
        mlflow.sklearn.autolog(log_models=False, max_tuning_runs=None, log_post_training_metrics=False, silent=True)
        print(f"Searching {len(search.param_grid['model__n_estimators']) * len(search.param_grid['model__max_depth']) * len(search.param_grid['model__min_samples_leaf'])} combinations with 5-fold CV...")
        search.fit(data["X_train"], data["y_train"])
        mlflow.sklearn.autolog(disable=True)

        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")
        log_child_test_metrics(search, param_grid, data)
        log_feature_importance(search.best_estimator_, "Random Forest - Grid Search")
        log_and_evaluate_model(search.best_estimator_, data, "Random Forest - Grid Search")


def register_candidates():
    """Register every candidate from this session and assign champion/challenger aliases"""
    banner("MODEL REGISTRY: REGISTER CANDIDATES AND PICK A CHAMPION")
    client = MlflowClient()

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.candidate = 'true' AND tags.training_session = '{SESSION_ID}'",
        order_by=["metrics.roc_auc DESC"],
    )
    print(runs[["tags.mlflow.runName", "metrics.roc_auc", "metrics.accuracy_score"]].to_string(index=False))

    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        pass  # already exists from a previous session
    client.update_registered_model(
        REGISTERED_MODEL_NAME,
        description="Titanic survival classifier. Use `models:/titanic-survival@champion` "
                    "to load the current best model.",
    )
    client.set_registered_model_tag(REGISTERED_MODEL_NAME, "task", "binary_classification")
    client.set_registered_model_tag(REGISTERED_MODEL_NAME, "owner", "mlflow-demo")

    versions = []
    for _, run in runs.iterrows():
        run_name = run["tags.mlflow.runName"]
        version = mlflow.register_model(f"runs:/{run.run_id}/model", REGISTERED_MODEL_NAME)
        client.update_model_version(
            REGISTERED_MODEL_NAME, version.version,
            description=f"**{run_name}** - test ROC-AUC {run['metrics.roc_auc']:.4f}, "
                        f"accuracy {run['metrics.accuracy_score']:.4f}",
        )
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "run_name", run_name)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "roc_auc", f"{run['metrics.roc_auc']:.4f}")
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "training_session", SESSION_ID)
        versions.append((version.version, run_name))
        print(f"Registered {run_name} as version {version.version}")

    # Runs are sorted by ROC-AUC, so the first two are the best
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", versions[0][0])
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", versions[1][0])
    print(f"\n@champion   -> version {versions[0][0]} ({versions[0][1]})")
    print(f"@challenger -> version {versions[1][0]} ({versions[1][1]})")


def run_champion_inference(data):
    """Load the champion by alias, re-validate it on the test set and score a few example passengers"""
    banner("INFERENCE: LOAD models:/titanic-survival@champion")
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    champion = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")

    passengers = pd.DataFrame([
        {"Name": "Rose (1st class, female, 17)",   "Pclass": 1, "Age": 17.0, "SibSp": 0, "Parch": 1, "Fare": 100.0, "Sex": "female"},
        {"Name": "Jack (3rd class, male, 20)",     "Pclass": 3, "Age": 20.0, "SibSp": 0, "Parch": 0, "Fare": 7.25,  "Sex": "male"},
        {"Name": "Cal (1st class, male, 30)",      "Pclass": 1, "Age": 30.0, "SibSp": 1, "Parch": 0, "Fare": 150.0, "Sex": "male"},
        {"Name": "Mother (3rd class, female, 35)", "Pclass": 3, "Age": 35.0, "SibSp": 1, "Parch": 3, "Fare": 20.0,  "Sex": "female"},
        {"Name": "Child (2nd class, male, 6)",     "Pclass": 2, "Age": 6.0,  "SibSp": 1, "Parch": 1, "Fare": 30.0,  "Sex": "male"},
        {"Name": "Unknown age (3rd class, male)",  "Pclass": 3, "Age": None, "SibSp": 0, "Parch": 0, "Fare": 8.05,  "Sex": "male"},
    ])

    with mlflow.start_run(run_name="Champion Inference"):
        mlflow.set_tag("mlflow.note.content",
            "Loads the registered model **by alias** (`models:/titanic-survival@champion`), re-validates it "
            "on the test set with `mlflow.evaluate`, and scores a few example passengers. "
            "See *Artifacts → predictions* for the results table."
        )
        mlflow.set_tag("training_session", SESSION_ID)
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("model_version", champion.version)
        mlflow.log_param("champion_run_name", champion.tags.get("run_name"))

        # Sanity-check the registered model before using it
        mlflow.evaluate(
            model_uri,
            data=data["test_dataset"],
            model_type="classifier",
            evaluator_config={"log_model_explainability": False},
        )

        model = mlflow.sklearn.load_model(model_uri)
        X = passengers[FEATURES]
        passengers["survival_probability"] = model.predict_proba(X)[:, 1].round(3)
        passengers["predicted_survived"] = model.predict(X)
        mlflow.log_table(passengers, "predictions/sample_passengers.json")
        mlflow.log_metric("mean_survival_probability", passengers["survival_probability"].mean())

        print(f"Champion: version {champion.version} ({champion.tags.get('run_name')})")
        print(passengers[["Name", "survival_probability", "predicted_survived"]].to_string(index=False))


def main():
    """Main training pipeline with MLflow experiments"""
    banner("TITANIC SURVIVAL PREDICTION - MLFLOW DEMO")

    # Setup MLflow
    setup_mlflow()

    df = load_dataset()
    data = split_data(df)

    # Run multiple experiments
    run_experiment_1_baseline(data)
    run_experiment_2_tuned_rf(data)
    run_experiment_3_logistic_regression(data)
    run_experiment_4_svm(data)
    run_experiment_5_gradient_boosting(data)
    run_experiment_6_grid_search(data)

    register_candidates()
    run_champion_inference(data)

    banner("ALL EXPERIMENTS COMPLETED!")
    print("\nView results in MLflow UI at: http://localhost:5000")
    print("\nKey MLflow features demonstrated:")
    print("  ✓ Experiment and run descriptions (markdown)")
    print("  ✓ Parameter and metric logging, incl. per-step learning curves")
    print("  ✓ Dataset tracking (mlflow.log_input)")
    print("  ✓ Model signatures and input examples")
    print("  ✓ Built-in evaluation plots (mlflow.evaluate)")
    print("  ✓ Autologging with nested runs for a grid search")
    print("  ✓ Model registry: versions, descriptions, tags, @champion/@challenger aliases")
    print("  ✓ Loading a model by alias for inference (mlflow.log_table)")
    print("=" * 80)


if __name__ == "__main__":
    main()
