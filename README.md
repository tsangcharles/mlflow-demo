# MLflow Model Management Demo# MLflow Model Management Demo - Titanic Survival Prediction# Titanic Survival Prediction - Docker Demo



A hands-on demonstration of **MLflow** for experiment tracking, model versioning, and ML lifecycle management using the Titanic survival prediction dataset. This project runs entirely in Docker and showcases industry-standard practices for managing machine learning experiments.



## 🎯 What You'll LearnA comprehensive demonstration of **MLflow** for model tracking, experiment management, and model registry using the Titanic dataset. This project showcases MLflow's key features in a fully Dockerized environment.A complete machine learning demo that trains a Random Forest classifier on the Titanic dataset and deploys it as a REST API using FastAPI and Docker.



This demo teaches practical MLflow skills:



- **Experiment Tracking** - Automatically log parameters, metrics, and artifacts## 🎯 Learning Objectives## 🎯 Project Overview

- **Model Registry** - Version and manage models throughout their lifecycle  

- **Run Comparison** - Compare multiple experiments side-by-side

- **Reproducibility** - Track everything needed to recreate any experiment

- **Production Workflows** - Use PostgreSQL backend and artifact storageThis demo teaches students:This project demonstrates:

- **Docker Integration** - Run MLflow in a containerized environment

1. **MLflow Experiment Tracking** - Track multiple ML experiments with different algorithms1. **Training**: Downloads the Titanic dataset from Kaggle and trains a machine learning model

## 🏗️ Architecture

2. **Parameter Logging** - Record hyperparameters for reproducibility2. **Deployment**: Serves the trained model via a FastAPI REST API

```

┌─────────────────┐3. **Metrics Logging** - Track model performance metrics (accuracy, precision, recall, F1, ROC-AUC)3. **Containerization**: Everything runs in Docker containers

│  PostgreSQL DB  │ ← Stores experiment metadata, parameters, metrics

└────────┬────────┘4. **Model Registry** - Register and version models

         │

┌────────▼────────┐5. **Artifact Management** - Store plots, reports, and model files## 📁 Project Structure

│  MLflow Server  │ ← Tracking server with web UI (http://localhost:5000)

└────────▲────────┘6. **Run Comparison** - Compare different model configurations side-by-side

         │

┌────────┴────────┐7. **Docker Integration** - Run MLflow in a production-like containerized environment```

│ Training Script │ ← Runs 4 experiments, logs everything to MLflow

└─────────────────┘supervised-learning-docker-demo/

```

## 🏗️ Architecture├── src/

**Three Docker Services:**

1. **PostgreSQL** - Persistent storage for MLflow metadata│   ├── train.py          # Model training script

2. **MLflow Server** - Central tracking server with web UI

3. **Training** - Runs ML experiments and logs to MLflow```│   └── api.py            # FastAPI application



## 📁 Project Structure┌─────────────────┐├── Dockerfile.train      # Docker image for training



```│  PostgreSQL DB  │  ← Stores experiment metadata├── Dockerfile.api        # Docker image for API

mlflow-demo/

├── src/└────────┬────────┘├── docker-compose.yml    # Orchestration configuration

│   └── train.py              # Training script with MLflow tracking

├── docker-compose.yml        # Multi-container orchestration         │├── requirements.txt      # Python dependencies

├── Dockerfile.train          # Training environment

├── Dockerfile.mlflow         # MLflow server environment  ┌────────▼────────┐└── README.md

├── requirements.txt          # Python dependencies

├── kaggle.json.template      # Template for Kaggle API credentials│  MLflow Server  │  ← Tracking server with UI (port 5000)```

└── README.md                 # This file

```└────────▲────────┘



## 🚀 Quick Start         │## 🚀 Quick Start



### Prerequisites┌────────┴────────┐



- Docker and Docker Compose installed│ Training Script │  ← Runs 4 experiments, logs to MLflow### Prerequisites

- Internet connection (downloads Titanic dataset from Kaggle)

- Kaggle account and API credentials (free)└─────────────────┘



### Step 1: Get Kaggle API Credentials```- Docker and Docker Compose installed



1. Create a free account at [kaggle.com](https://www.kaggle.com)- Internet connection (to download dataset from Kaggle)

2. Go to **Account Settings** → **API** → **Create New API Token**

3. This downloads `kaggle.json` containing your credentials### Services:- Kaggle API credentials (free Kaggle account required)

4. **Place `kaggle.json` in the project root directory**

- **PostgreSQL**: Backend store for MLflow metadata (runs, params, metrics)

⚠️ **Important**: Never commit `kaggle.json` to version control (already in `.gitignore`)

- **MLflow Server**: Central tracking server with web UI### Step 1: Set Up Kaggle API Credentials

### Step 2: Start the Demo

- **Training Container**: Runs multiple ML experiments and logs to MLflow

```bash

docker compose up1. Create a free account at [kaggle.com](https://www.kaggle.com)

```

## 📁 Project Structure2. Go to your account settings: https://www.kaggle.com/settings

This command will:

1. ✅ Start PostgreSQL database3. Scroll down to the "API" section

2. ✅ Start MLflow tracking server  

3. ✅ Download Titanic dataset from Kaggle```4. Click "Create New API Token" - this downloads `kaggle.json`

4. ✅ Run 4 machine learning experiments:

   - Random Forest (baseline)mlflow-demo/5. Place the `kaggle.json` file in this project's root directory

   - Random Forest (tuned hyperparameters)

   - Logistic Regression├── src/

   - Support Vector Machine (SVM)

5. ✅ Log all results to MLflow│   └── train.py              # Training script with MLflow tracking**Important**: The `kaggle.json` file contains your credentials. Do NOT commit it to git (it's already in `.gitignore`).



**Wait time**: ~3-5 minutes for all experiments to complete├── docker-compose.yml        # Multi-container orchestration



### Step 3: Explore Results├── Dockerfile.train          # Training environment image**Note**: A `kaggle.json.template` file is provided as a reference for the expected format.



Open your browser and go to:├── requirements.txt          # Python dependencies



```├── kaggle.json.template      # Template for Kaggle credentials### Step 2: Run the Demo

http://localhost:5000

```└── README.md



You'll see the **MLflow UI** with:```Simply run:

- Experiment tracking dashboard

- 4 completed runs with metrics

- Model registry with versioned models

- Plots and artifacts for each run## 🚀 Quick Start```bash



## 📊 What Gets Loggeddocker compose up



For each of the 4 experiments, MLflow tracks:### Prerequisites```



### Parameters

- All hyperparameters (e.g., `n_estimators`, `max_depth`, `C`, `kernel`)

- Model type and configuration- Docker and Docker Compose installedThis single command will:

- Dataset split sizes

- Internet connection (to download Titanic dataset from Kaggle)1. Build both Docker images (training and API)

### Metrics

- **Accuracy** - Overall prediction correctness- Kaggle API credentials (free account)2. Download the Titanic dataset from Kaggle

- **Precision** - Of predicted survivors, % actually survived

- **Recall** - Of actual survivors, % correctly identified  3. Train the Random Forest model

- **F1 Score** - Harmonic mean of precision and recall

- **ROC-AUC** - Area under the ROC curve### Step 1: Set Up Kaggle Credentials4. Save the model as a pickle file



### Artifacts5. Start the FastAPI server with the trained model

- **Feature importance plot** - Which features matter most

- **Confusion matrix** - Visual breakdown of predictions1. Create a free account at [kaggle.com](https://www.kaggle.com)

- **Classification report** - Detailed performance metrics

- **Trained model** - Serialized scikit-learn model2. Go to Settings → API → "Create New API Token"The API will be available at `http://localhost:8000`



### Metadata3. This downloads `kaggle.json`

- Timestamp of each run

- Dataset information4. Place `kaggle.json` in the project root directory## 📊 Model Details

- Task type (binary classification)



## 🔬 The Four Experiments

**Security Note**: Never commit `kaggle.json` to version control!### Features Used

### Experiment 1: Random Forest (Baseline)

```python- **Pclass**: Passenger class (1st, 2nd, 3rd)

RandomForestClassifier(

    n_estimators=100,### Step 2: Launch the Demo- **Sex**: Gender (male/female)

    max_depth=None,

    min_samples_split=2,- **Age**: Age in years

    min_samples_leaf=1,

    random_state=42```bash- **SibSp**: Number of siblings/spouses aboard

)

```docker compose up- **Parch**: Number of parents/children aboard

**Purpose**: Establish baseline performance

```- **Fare**: Passenger fare

### Experiment 2: Random Forest (Tuned)

```python

RandomForestClassifier(

    n_estimators=200,This command will:### Target Variable

    max_depth=10,

    min_samples_split=5,1. ✅ Start PostgreSQL database for MLflow- **Survived**: 0 = Did not survive, 1 = Survived

    min_samples_leaf=2,

    random_state=422. ✅ Launch MLflow tracking server

)

```3. ✅ Download Titanic dataset### Algorithm

**Purpose**: Show impact of hyperparameter tuning

4. ✅ Run 4 different ML experiments:- Random Forest Classifier with 100 estimators

### Experiment 3: Logistic Regression

```python   - Random Forest (baseline)

LogisticRegression(

    penalty='l2',   - Random Forest (tuned)## 🔌 API Usage

    C=1.0,

    solver='lbfgs',   - Logistic Regression

    max_iter=1000,

    random_state=42   - Support Vector Machine### Interactive Documentation

)

```5. ✅ Log all results to MLflow

**Purpose**: Compare simpler linear model

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

### Experiment 4: Support Vector Machine

```python### Step 3: Explore MLflow UI

SVC(

    kernel='rbf',### Endpoints

    C=1.0,

    gamma='scale',Open your browser and navigate to:

    probability=True,

    random_state=42```#### 1. Health Check

)

```http://localhost:5000```bash

**Purpose**: Compare kernel-based approach

```curl http://localhost:8000/health

## 🎓 Using the MLflow UI

```

### View All Experiments

You'll see the **MLflow UI** with all tracked experiments!

1. Navigate to http://localhost:5000

2. Click on **"titanic-survival-prediction"** experiment#### 2. Single Prediction

3. See table of all 4 runs with their metrics

## 📊 What Gets Logged to MLflow```bash

### Compare Multiple Runs

curl -X POST "http://localhost:8000/predict" \

1. Select 2 or more runs (checkboxes)

2. Click **"Compare"** buttonFor each of the 4 experiments, the training script logs:  -H "Content-Type: application/json" \

3. View side-by-side:

   - Parameter differences  -d '{

   - Metric comparisons  

   - Visualization plots### 1. **Parameters**    "pclass": 3,



**Try This**: Compare baseline vs tuned Random Forest to see impact of hyperparameters- Algorithm hyperparameters (n_estimators, max_depth, C, kernel, etc.)    "sex": "male",



### Explore a Single Run- Model type    "age": 22.0,



1. Click on any run name- Train/test split sizes    "sibsp": 1,

2. Tabs available:

   - **Overview** - Run summary and metadata    "parch": 0,

   - **Parameters** - All hyperparameters used

   - **Metrics** - Performance metrics with charts### 2. **Metrics**    "fare": 7.25

   - **Artifacts** - Plots, models, and reports

- Accuracy  }'

**Try This**: Click on a run → Artifacts → `plots/feature_importance.png` to see which features matter most

- Precision```

### Use the Model Registry

- Recall

1. Click **"Models"** tab in top navigation

2. See all registered models:- F1 ScoreExample response:

   - `titanic_random_forest_baseline`

   - `titanic_random_forest_tuned`- ROC-AUC (when applicable)```json

   - `titanic_logistic_regression`

   - `titanic_svm`{

3. Click a model to see versions and stages

### 3. **Artifacts**  "survived": 0,

**Try This**: Promote the best model to "Staging" or "Production" stage

- Trained model (sklearn format)  "probability": 0.23,

## 💡 Understanding the Results

- Feature importance plot  "prediction_text": "Did not survive"

### Expected Performance (Approximate)

- Confusion matrix plot}

| Model | Accuracy | Precision | Recall | F1 Score |

|-------|----------|-----------|--------|----------|- Classification report (text)```

| Random Forest (Baseline) | ~0.82 | ~0.81 | ~0.72 | ~0.76 |

| Random Forest (Tuned) | ~0.82 | ~0.81 | ~0.73 | ~0.77 |

| Logistic Regression | ~0.80 | ~0.79 | ~0.70 | ~0.74 |

| SVM | ~0.81 | ~0.80 | ~0.71 | ~0.75 |### 4. **Tags**#### 3. Batch Prediction



*Note: Values may vary slightly due to data preprocessing*- Dataset name```bash



### Key Insights- Task type (binary classification)curl -X POST "http://localhost:8000/predict/batch" \



**Most Important Features** (typically):  -H "Content-Type: application/json" \

1. **Sex** - Gender strongly predicts survival

2. **Pclass** - Passenger class matters (1st > 2nd > 3rd)### 5. **Model Registry**  -d '[

3. **Fare** - Ticket price correlates with survival

- Each model is registered with a unique name    {

**Model Comparison**:

- Random Forest models perform best overall- Enables model versioning and lifecycle management      "pclass": 1,

- Hyperparameter tuning shows modest improvement

- All models beat random guessing (~50% accuracy)      "sex": "female",



## 🛠️ Useful Commands## 🔬 Experiments Included      "age": 28.0,



### View Logs      "sibsp": 0,



```bash### Experiment 1: Random Forest (Baseline)      "parch": 0,

# Watch training progress

docker compose logs -f train```python      "fare": 100.0



# Check MLflow servern_estimators = 100    },

docker compose logs mlflow

max_depth = None    {

# Check database

docker compose logs postgresmin_samples_split = 2      "pclass": 3,



# View all logsmin_samples_leaf = 1      "sex": "male",

docker compose logs

``````      "age": 22.0,



### Container Management      "sibsp": 1,



```bash### Experiment 2: Random Forest (Tuned)      "parch": 0,

# Check status

docker compose ps```python      "fare": 7.25



# Stop everythingn_estimators = 200    }

docker compose down

max_depth = 10  ]'

# Fresh start (removes all data)

docker compose down -vmin_samples_split = 5```

docker compose up --build

min_samples_leaf = 2

# Run in background

docker compose up -d```## 🧪 Example Predictions

```



### Access Services

### Experiment 3: Logistic Regression### High Survival Probability

```bash

# MLflow UI```pythonFirst-class female passenger:

http://localhost:5000

penalty = 'l2'```json

# PostgreSQL (if needed)

Host: localhostC = 1.0{

Port: 5432

Database: mlflowsolver = 'lbfgs'  "pclass": 1,

User: mlflow

Password: mlflowmax_iter = 1000  "sex": "female",

```

```  "age": 28,

## 🔧 Customization

  "sibsp": 0,

### Add Your Own Experiment

### Experiment 4: Support Vector Machine  "parch": 0,

Edit `src/train.py` and add a new function:

```python  "fare": 100.0

```python

def run_experiment_5_custom(X_train, X_test, y_train, y_test):kernel = 'rbf'}

    """Experiment 5: Your Custom Model"""

    model = RandomForestClassifier(C = 1.0```

        n_estimators=150,

        max_depth=15,gamma = 'scale'

        min_samples_split=3,

        random_state=42probability = True### Low Survival Probability

    )

    params = {```Third-class male passenger:

        "n_estimators": 150,

        "max_depth": 15,```json

        "min_samples_split": 3,

        "random_state": 42## 🎓 Teaching Guide - MLflow Features{

    }

      "pclass": 3,

    train_and_log_model(

        X_train, X_test, y_train, y_test, ### 1. **Experiment Tracking**  "sex": "male",

        model, "Random Forest - Custom", params

    )  "age": 22,

```

Navigate to the MLflow UI and show students:  "sibsp": 1,

Then call it in `main()`:

- Click on "titanic-survival-prediction" experiment  "parch": 0,

```python

def main():- See all 4 runs listed with their metrics  "fare": 7.25

    # ... existing code ...

    run_experiment_5_custom(X_train, X_test, y_train, y_test)- Click on any run to see detailed information}

```

```

Rebuild and run:

**Teaching Point**: MLflow automatically tracks when experiments were run, by whom, and with what parameters.

```bash

docker compose down -v## 🛠️ Development

docker compose up --build

```### 2. **Comparing Runs**



### Try Different Algorithms### View Training Logs



Add experiments with:In the MLflow UI:```bash

- XGBoost

- Neural Networks (Keras/PyTorch)1. Select multiple runs (checkbox each run)docker compose logs train

- Gradient Boosting

- Ensemble methods2. Click "Compare" button```



### Experiment with Features3. View side-by-side comparison of:



Modify `preprocess_data()` to:   - Parameters### View API Logs

- Add feature engineering (e.g., family size = SibSp + Parch)

- Try different encoding strategies   - Metrics```bash

- Test feature selection techniques

   - Visualizationsdocker compose logs api

## 🐛 Troubleshooting

```

### Issue: Containers won't start

**Teaching Point**: Easy comparison helps identify the best performing model and understand impact of hyperparameters.

**Solution**:

```bash### Rebuild Everything

docker compose down -v

docker compose up --build### 3. **Visualizing Metrics**```bash

```

docker compose down -v

### Issue: "Kaggle authentication failed"

In the MLflow UI:docker compose up --build

**Solution**:

- Verify `kaggle.json` is in the project root1. Click on a specific run```

- Check file format is correct:

  ```json2. Scroll to "Artifacts" section

  {"username":"your_username","key":"your_api_key"}

  ```3. Open "plots" folder### Stop Services

- Re-download from https://www.kaggle.com/settings

4. View feature importance and confusion matrix plots```bash

### Issue: MLflow UI shows no experiments

docker compose down

**Solution**:

- Wait for training to complete (~5 minutes)**Teaching Point**: MLflow stores any artifact (plots, data files, models) alongside experiment metadata.```

- Check logs: `docker compose logs train`

- Look for "ALL EXPERIMENTS COMPLETED!" message

- Refresh browser

### 4. **Model Registry**## 📚 Teaching Notes

### Issue: Training keeps retrying MLflow connection



**Solution**:

- Normal during startup - wait up to 60 secondsIn the MLflow UI:This demo is ideal for teaching:

- If it fails after 12 attempts:

  ```bash1. Click "Models" tab at the top

  docker compose down -v

  docker compose up --build2. See all registered models:1. **Machine Learning Pipeline**

  ```

   - `titanic_random_forest_baseline`   - Data loading and preprocessing

### Issue: Port 5000 already in use

   - `titanic_random_forest_tuned`   - Feature engineering (encoding categorical variables)

**Solution**: Change port in `docker-compose.yml`:

```yaml   - `titanic_logistic_regression`   - Model training and evaluation

mlflow:

  ports:   - `titanic_svm`   - Model serialization (pickle)

    - "5001:5000"  # Use port 5001 instead

```3. Click on a model to see versions and stages

Then access UI at http://localhost:5001

2. **API Development**

## 📚 Learning Resources

**Teaching Point**: Model registry enables model versioning, staging (dev/staging/production), and lifecycle management.   - RESTful API design

### MLflow Documentation

- [Official Documentation](https://mlflow.org/docs/latest/)   - Input validation with Pydantic

- [Tracking Quickstart](https://mlflow.org/docs/latest/tracking.html)

- [Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)### 5. **Reproducibility**   - Error handling



### Related Topics   - API documentation with FastAPI

- [Docker Compose](https://docs.docker.com/compose/)

- [scikit-learn](https://scikit-learn.org/stable/)Show students how to reproduce an experiment:

- [Pandas](https://pandas.pydata.org/docs/)

1. Click on any run in MLflow UI3. **DevOps & Docker**

### Titanic Dataset

- [Kaggle Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)2. View all parameters logged   - Multi-stage containerization

- [Titanic Challenge](https://www.kaggle.com/c/titanic)

3. See the exact command and environment used   - Service orchestration with Docker Compose

## 🎯 Why MLflow Matters

   - Volume management for model persistence

### Common ML Problems

**Teaching Point**: MLflow ensures experiments are reproducible by tracking all relevant information.   - Service dependencies

❌ **"Which hyperparameters did I use for that great result?"**  

✅ MLflow logs all parameters automatically



❌ **"How do I compare 50 different experiments?"**  ## 🛠️ Useful Commands4. **Data Science Best Practices**

✅ MLflow provides comparison UI and programmatic access

   - Handling missing values

❌ **"Where is the model I trained last week?"**  

✅ MLflow Model Registry with versioning### View Logs   - Train-test split



❌ **"Can I reproduce this experiment?"**  ```bash   - Model evaluation metrics

✅ MLflow tracks code, data, environment, and configuration

# Training logs   - Feature importance analysis

❌ **"Which model is currently in production?"**  

✅ MLflow Registry tracks model stages and lineagedocker compose logs train



### Industry Usage## 🔍 Understanding the Workflow



MLflow is used by:# MLflow server logs

- **Databricks** - Created and maintains MLflow

- **Netflix** - Experiment tracking at scaledocker compose logs mlflow1. **Training Container** (`train` service):

- **Microsoft** - Integration with Azure ML

- **Uber** - ML platform foundation   - Downloads Titanic dataset from Kaggle using `kagglehub`

- Many startups and enterprises

# PostgreSQL logs   - Preprocesses data (handles missing values, encodes categorical features)

## 🎓 Teaching Notes

docker compose logs postgres   - Trains Random Forest model

### For Instructors

```   - Saves model and label encoder to shared volume

This demo is designed for:

- **Data Science courses** - ML lifecycle and experiment management

- **MLOps workshops** - Production ML practices

- **Bootcamps** - Hands-on industry tools### Restart Everything2. **API Container** (`api` service):

- **Self-learners** - Practical MLflow experience

```bash   - Waits for training to complete

**Suggested Flow** (60 minutes):

1. **Introduction** (10 min) - Explain MLflow and why it mattersdocker compose down -v   - Loads trained model from shared volume

2. **Demo** (10 min) - Run `docker compose up` and explain services

3. **Exploration** (20 min) - Navigate MLflow UI togetherdocker compose up --build   - Starts FastAPI server

4. **Hands-on** (15 min) - Students modify and run experiments

5. **Discussion** (5 min) - Q&A and key takeaways```   - Serves predictions via REST API



### For Students



**Learning Objectives**:### Keep MLflow UI Running After Training3. **Shared Volume**:

- Understand experiment tracking workflow

- Use MLflow UI to compare modelsBy default, the training container exits after completion, but MLflow UI stays running. To view results:   - Docker volume `models` persists trained model

- Interpret ML metrics and visualizations

- Apply MLflow to your own projects```bash   - Shared between training and API containers



**Practice Exercises**:# Training completes, but you can still browse at http://localhost:5000

1. Run the demo and explore all 4 experiments

2. Add a 5th experiment with your own parameters# Press Ctrl+C when done exploring## 📦 Dataset Source

3. Compare all runs and identify the best model

4. Analyze feature importance - why do certain features matter?

5. Export the best model and document your findings

# Or run in detached modeThe Titanic dataset is automatically downloaded from Kaggle:

## 📝 Assignment Ideas

docker compose up -d- **Source**: https://www.kaggle.com/datasets/yasserh/titanic-dataset

### Beginner

- Run all experiments and create a comparison report```- **Method**: Official Kaggle API Python package

- Identify which hyperparameters most impact performance

- Explain the confusion matrix for the best model- **Authentication**: Requires free Kaggle account and API token



### Intermediate  ### Stop All Services

- Add 3 new experiments with different algorithms

- Implement cross-validation and log results```bash## 💡 Tips for Students

- Create custom visualizations as artifacts

docker compose down

### Advanced

- Integrate hyperparameter tuning (GridSearch/RandomSearch)```1. Try modifying the model parameters in `src/train.py`

- Add automated model selection based on metrics

- Implement MLflow model serving2. Add new features to improve accuracy

- Deploy the best model as a web service

### Remove All Data (Fresh Start)3. Experiment with different algorithms (SVM, Logistic Regression, etc.)

## 🤝 Contributing

```bash4. Add more API endpoints (e.g., model metrics, feature importance)

This is an educational demo. Suggested improvements:

- Additional algorithms (XGBoost, LightGBM, Neural Networks)docker compose down -v5. Implement model versioning

- Hyperparameter tuning integration

- Cross-validation experiments```6. Add data validation and error handling

- Model explainability (SHAP, LIME)

- Automated testing



## 📄 License## 📚 Key MLflow Concepts Demonstrated## 🐛 Troubleshooting



Educational demo for teaching purposes. Feel free to use and modify for learning.



## 🙏 Acknowledgments### 1. **Tracking Server****Issue**: `FileNotFoundError: kaggle.json`



- **Titanic Dataset**: [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)- Centralized server for storing experiment data- **Solution**: Make sure you've downloaded your `kaggle.json` file from Kaggle and placed it in the project root directory. See "Step 1: Set Up Kaggle API Credentials" above.

- **MLflow**: [mlflow.org](https://mlflow.org/) - Open source ML lifecycle platform

- **scikit-learn**: [scikit-learn.org](https://scikit-learn.org/) - Machine learning in Python- Web UI for visualization and comparison

- **Docker**: [docker.com](https://www.docker.com/) - Containerization platform

- API for programmatic access**Issue**: API returns 503 "Model not loaded"

---

- **Solution**: Ensure training completed successfully. Check logs with `docker compose logs train`

**Ready to start tracking your ML experiments? Run `docker compose up` and explore! 🚀**

### 2. **Backend Store (PostgreSQL)**

- Stores experiment metadata (params, metrics, tags)**Issue**: Dataset download fails or "401 Unauthorized"

- Enables querying and filtering experiments- **Solution**: Verify your `kaggle.json` file is valid. Try re-downloading it from https://www.kaggle.com/settings

- Production-ready database backend

**Issue**: Port 8000 already in use

### 3. **Artifact Store**- **Solution**: Stop other services using port 8000 or modify the port in `docker-compose.yml`

- Stores large files (models, plots, datasets)

- Organized by run ID## 📝 License

- Supports multiple storage backends (local, S3, Azure, GCS)

This is an educational demo for teaching purposes.

### 4. **Experiments & Runs**

- **Experiment**: A collection of related runs (e.g., "titanic-survival-prediction")## 🙏 Acknowledgments

- **Run**: A single execution of training code with specific parameters

- Titanic dataset from Kaggle

### 5. **Model Registry**- FastAPI framework

- Centralized model store- scikit-learn library

- Model versioning (v1, v2, v3...)

- Model stages (None, Staging, Production, Archived)
- Model lineage and metadata

## 🔧 Customization Ideas for Students

1. **Add More Experiments**
   - Try different algorithms (XGBoost, Neural Networks)
   - Experiment with feature engineering
   - Test different train/test splits

2. **Hyperparameter Tuning**
   - Implement grid search or random search
   - Log all tuning iterations to MLflow
   - Compare tuning approaches

3. **Model Deployment**
   - Use MLflow's built-in model serving
   - Deploy best model as REST API
   - Compare with traditional Flask/FastAPI deployment

4. **Advanced Metrics**
   - Log learning curves
   - Track training time
   - Monitor cross-validation scores

5. **Model Comparison Dashboard**
   - Create custom visualizations
   - Export comparison reports
   - Automate model selection based on metrics

## 🐛 Troubleshooting

### Issue: MLflow UI shows no experiments
- **Solution**: Wait for training to complete, check `docker compose logs train`

### Issue: Cannot connect to MLflow server
- **Solution**: Ensure MLflow container is healthy: `docker compose ps`

### Issue: PostgreSQL connection error
- **Solution**: Wait for PostgreSQL to be ready (health check takes ~10 seconds)

### Issue: Kaggle authentication failed
- **Solution**: Verify `kaggle.json` is in the root directory with correct credentials

### Issue: Port 5000 already in use
- **Solution**: Change MLflow port in `docker-compose.yml`:
  ```yaml
  ports:
    - "5001:5000"  # Use port 5001 instead
  ```

## 🎯 Assessment Ideas for Students

1. **Run Analysis**: Which model performed best? Why?
2. **Parameter Impact**: How does `max_depth` affect Random Forest performance?
3. **Feature Engineering**: Can you improve accuracy by adding new features?
4. **Visualization**: Create a comparison chart of all model accuracies
5. **Model Selection**: Which model would you deploy to production? Justify your choice.

## 📖 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)

## 🎓 Why MLflow Matters

### Problems in ML Development:
- ❌ "Which parameters did I use for that good result?"
- ❌ "How do I compare 50 different experiments?"
- ❌ "Where is the model I trained last week?"
- ❌ "Can I reproduce this experiment?"
- ❌ "How do I track which model is in production?"

### MLflow Solutions:
- ✅ Automatic parameter tracking
- ✅ Built-in experiment comparison UI
- ✅ Centralized model registry with versioning
- ✅ Complete experiment reproducibility
- ✅ Model lifecycle management (staging, production)

## 📝 License

Educational demo for teaching purposes.

## 🙏 Acknowledgments

- Titanic dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- [MLflow](https://mlflow.org/) - Open source platform for ML lifecycle
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
