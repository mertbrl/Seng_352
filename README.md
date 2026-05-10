# Metro Interstate Traffic Volume Prediction

This project predicts hourly `traffic_volume` for the Metro Interstate Traffic Volume dataset using time-based and weather-related features. It is designed as a clean student ML project: notebooks explain the work, reusable logic lives in `src/`, and the final model is used by a small Streamlit web app.

## Project Structure

```text
Traffic_Volume_Prediction/
|-- app/
|   `-- app.py                         # Streamlit prediction dashboard
|-- config/
|   |-- paths.yaml                     # dataset, model, and report paths
|   `-- params.yaml                    # split ratios, preprocessing, models, tuning
|-- data/
|   |-- raw/                           # original CSV
|   `-- processed/                     # optional processed outputs
|-- models/
|   |-- final_model.joblib             # saved final regressor
|   `-- preprocessing_artifacts.joblib # scaler, selected features, encoding metadata
|-- notebooks/
|   |-- 01_eda.ipynb                   # EDA and visualization notebook
|   `-- 02_preprocessing_and_modeling.ipynb
|-- reports/
|   |-- figures/                       # exported reference figures
|   |-- tables/                        # metrics, importance, errors, ablation
|   `-- model_card.md
|-- src/
|   |-- preprocessing.py               # reusable preprocessing pipeline
|   |-- modeling.py                    # reusable model comparison/evaluation logic
|   `-- inference.py                   # app-ready prediction helpers and weather API logic
|-- run_train.py                       # reproducible end-to-end training script
|-- requirements.txt
`-- README.md
```

## Dataset

The expected dataset path is:

```text
data/raw/Metro_Interstate_Traffic_Volume.csv
```

The target is `traffic_volume`, so this is a regression task.

## Preprocessing Pipeline

The preprocessing pipeline follows a leakage-safe chronological workflow:

- Load the CSV, parse `date_time` as datetime, and sort records chronologically.
- Remove only physically invalid records before splitting: rows where `temp == 0`.
- Split the cleaned data chronologically into 70% training, 15% validation, and 15% test sets.
- Learn the rainfall cap from the training set only, then apply the same cap to validation and test.
- Engineer raw time features: `hour`, `day_of_week`, `month`, and `year`.
- Engineer indicator features: `is_weekend`, `is_rush_hour`, `is_night`, and `is_holiday`.
- Add cyclical sine/cosine encodings for hour, weekday, and month.
- Engineer weather features: `temp_c`, `temp_c2`, `rain_log`, `snow_log`, `is_raining`, `is_snowing`, and `clouds_all_pct`.
- Compute `rain_log` after rainfall capping.
- Fit Isolation Forest on the training set only.
- Remove detected outliers only from the training set; validation and test remain unchanged.
- Learn rare `weather_main` categories from the training set only.
- Group rare weather categories into `Other`.
- One-hot encode grouped weather categories separately for each split.
- Align validation and test encoded columns to the training-defined column structure.
- Separate features and target into `X` and `y`.
- Fit RobustScaler on training numeric features only.
- Transform validation and test using the fitted scaler.
- Select features using Random Forest feature importance fitted on the training set only.
- Run top-k feature ablation for `10`, `15`, `20`, and `25` selected features.
- Select the final feature count based on validation performance.

The ablation results support `top_k = 10` as the final selected feature count.

Short notebook version:

The data is loaded, `date_time` is parsed, and records are sorted chronologically. Only physically invalid `temp == 0` rows are removed before a 70/15/15 chronological train-validation-test split. All learned preprocessing steps, including rainfall capping, rare weather grouping, outlier detection, scaling, and Random Forest feature selection, are fitted on the training set only. Validation and test data are transformed with the training-fitted artifacts, and encoded columns are aligned to the training column structure. Top-k ablation over `10`, `15`, `20`, and `25` features selected `top_k = 10` based on validation performance.

Presentation version:

- Chronological 70/15/15 split; no random split.
- Only `temp == 0` is removed before splitting as invalid-record cleaning.
- Rainfall capping, rare-category grouping, Isolation Forest, scaling, and feature selection are fitted on training data only.
- Top-k ablation selected `top_k = 10` using validation performance.

## Modeling Pipeline

Model selection uses validation-set comparison instead of random K-fold cross-validation because the dataset is time ordered.

Baseline regressors compared:

- Linear Regression
- Ridge
- Lasso
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- SVR
- KNN Regressor
- MLP Regressor

Additional modeling steps:

- Hyperparameter tuning for the best validation candidates.
- Validation-aware tuning with `PredefinedSplit`.
- Voting Regressor evaluation.
- Stacking Regressor evaluation.
- Final test evaluation only after model selection.
- Seed sensitivity check for Random Forest.
- Error analysis on highest-error test examples.
- Feature selection importance and permutation importance.

Final selected model:

```text
Tuned Extra Trees Regressor
```

Final test metrics:

```text
MAE  = 261.91
RMSE = 485.85
R2   = 0.940
```

## Evaluation Metrics

Regression metrics are used:

- MAE
- RMSE
- R2

Classification metrics such as accuracy, F1, and ROC-AUC are not used because the task is regression.

## Notebooks

`notebooks/01_eda.ipynb` contains the EDA work directly in notebook cells. The visualizations are produced by notebook code and shown under the cells, including:

- Missingness heatmap
- Univariate histograms and KDE plots
- Skewness and kurtosis table
- Correlation heatmap
- Outlier boxplots
- Time feature plots
- Hour-weekday heatmap
- Weather category plots
- Pair plot
- Parallel coordinates
- PCA projection
- t-SNE projection
- Target-conditional distributions
- Train-test distribution comparison
- Daily traffic trend

`notebooks/02_preprocessing_and_modeling.ipynb` shows the modeling work directly in notebook cells:

- Pipeline summary
- Feature count ablation
- Selected features
- Feature selection importance
- Baseline model comparison
- Tuned model results
- Ensemble results
- Final test evaluation
- Permutation importance
- Error analysis
- Seed sensitivity

## Reports and Saved Outputs

Training saves:

- `models/final_model.joblib`
- `models/preprocessing_artifacts.joblib`
- `reports/tables/model_metrics.csv`
- `reports/tables/feature_count_ablation.csv`
- `reports/tables/feature_selection_importance.csv`
- `reports/tables/permutation_importance.csv`
- `reports/tables/highest_test_errors.csv`
- `reports/tables/seed_sensitivity.csv`
- `reports/model_card.md`

## Web App and Weather API

The app is implemented in:

```text
app/app.py
```

The user enters:

- date
- hour

The app then:

1. Fetches weather from the Open-Meteo API for the Minneapolis I-94 area.
2. Uses fallback weather values if API weather is unavailable.
3. Creates the same model input format used during training.
4. Loads the saved preprocessing artifacts.
5. Loads the saved final model.
6. Predicts traffic volume.
7. Displays traffic severity: Low, Moderate, or High.
8. Shows a 24-hour prediction preview.

The weather API helper functions are in:

```text
src/inference.py
```

Run the app:

```bash
streamlit run app/app.py
```

If `streamlit` is not on PATH:

```powershell
C:\Users\MertBrsl\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run app\app.py
```

## Reproducibility

The project uses:

- Config files for paths and parameters.
- Fixed random seed.
- Chronological split.
- Train-only fitted preprocessing transformations.
- Saved model and preprocessing artifacts.
- Saved metrics and report tables.

Run training:

```bash
pip install -r requirements.txt
python run_train.py
```
