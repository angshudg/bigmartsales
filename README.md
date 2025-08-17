# Big Mart Sales Prediction

This project predicts the number of units sold for items in various outlets using machine learning.  
The task is based on the Big Mart Sales dataset, where the target variable is `Item_Outlet_Sales`.
---

## Project Structure

```
.
├── dev_train.py     # Developmental script for experimentation & feature engineering
├── train.py         # Final training pipeline, trains model and saves it
├── inference.py     # Inference pipeline, loads trained model and scores test data
├── data/            # Folder to keep raw train/test data
├── models/          # Saved models
├── output/          # Scored test files
└── README.md        # Project documentation
```

---

## Workflow

### 1. Data Preparation
- Input data columns (train/test):  
  ```
  Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type,
  Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size,
  Outlet_Location_Type, Outlet_Type, Item_Outlet_Sales (train only)
  ```

### 2. Handling Missing Values
- `Item_Weight` → Predicted using other **item-level features**.  
- `Outlet_Size` → Predicted using other **outlet-level features**.  
- Auxiliary regressors/classifiers are trained to impute missing values before training the final model.

### 3. Model Training
- Main task: Predict `Item_Units_Sold`.  
- Model used: CatBoostRegressor with categorical feature support.  
- Training code is in `train.py`, which:
  - Reads the training data.
  - Imputes missing values using the auxiliary models.
  - Trains the CatBoost model on complete data.
  - Saves the trained model (`models/catboost_model.cbm`).

### 4. Inference
- `inference.py`:
  - Loads test data (same format as train, without `Item_Outlet_Sales`).
  - Imputes missing values using the same auxiliary models.
  - Applies the trained CatBoost model to predict `Item_Outlet_Sales`:
  - Stores results at a specified location (`output/scored_test.csv`).

---

## Running the Code

### Train the Model
```bash
python train.py --train_file data/train_v9rqX0R.csv --model_dir models/catboost_model.cbm
```

### Run Inference
```bash
python inference.py --test_file data/test_AbJTz2l.csv --model_dir models/catboost_model.cbm --submission_file output/scored_test.csv
```

---

## Development
- `dev_train.py` contains **developmental code** for:
  - Exploratory Data Analysis (EDA).
  - Comparing models (XGBoost, RandomForest, CatBoost, etc.).
  - Cross-validation experiments.
  - Feature selection and testing whether to include imputed features.

This script is **not used in production**, but serves as a playground for iterative model improvements.

---

## Notes
- Ensure all categorical variables are properly handled (CatBoost accepts raw string categories).  
- Missing value imputations are done using auxiliary models trained on subsets of features.  
- Including imputed columns in the final model is optional — their value is validated via cross-validation.

---

## Requirements

### Pip installation
```bash
pip install -r requirements.txt
```

### Conda environment
```bash
conda create -n bigmart python=3.9
conda activate bigmart
pip install -r requirements.txt
```

---

## requirements.txt contents

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
catboost>=1.2
xgboost>=1.7
lightgbm>=3.3
matplotlib>=3.6
seaborn>=0.12
joblib>=1.2
```

---
