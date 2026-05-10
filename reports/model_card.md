# Model Card: Metro Traffic Volume Predictor

## Purpose
Predict hourly `traffic_volume` using time and weather features.

## Validation Strategy
The project uses a chronological 70/15/15 train-validation-test split. Model comparison uses the validation set, not random K-fold CV.

## Final Model
Extra Trees Tuned

## Test Metrics
- MAE: 261.911
- RMSE: 485.846
- R2: 0.940

## Selected Features
hour_cos, hour, hour_sin, dow_sin, day_of_week, is_weekend, is_night, month_cos, temp_c, temp

## Notes
Rainfall capping, rare category grouping, scaling, and feature selection are fitted on training data only.
