# machine-learning-for-longitudinal-data

R scripts for simulation studies and real-world panel data (`nlswork`) evaluation comparing statistical and machine learning methods for longitudinal data, including **gee**, **geeb**, **reemforest**, **merf**, and **gpboost**[cite: 2, 3, 5, 6, 7, 8].

---

## Methods & Evaluation Metrics

### Compared Methods
* **gee** (`geepack::geeglm`): Generalized Estimating Equations supporting working correlation structures such as `independence`, `ar1`, and `unstructured` (`corstr`)[cite: 2, 3, 5, 6, 7, 8].
* **geeb** (custom `geebm()`, `predict.geebm()`): Gradient boosting framework for longitudinal data that iteratively updates residuals and estimates correlation via `geeglm` (default: `iteration = 100`, `feature_rate = 1.0`, `lrate = 0.1`)[cite: 2, 3, 5, 6, 7, 8].
* **reemforest** (`LongituRF::REEMforest`): Random Effects EM Forest (`sto = "none"`)[cite: 2, 3, 5, 6, 7, 8].
* **merf** (`LongituRF::MERF`): Mixed Effects Random Forest (`sto = "none"`)[cite: 2, 3, 5, 6, 7, 8].
* **gpboost** (`gpboost::gpboost` + `GPModel`): Tree-boosting combined with grouped random effects / Gaussian processes (default: `nrounds = 200`, `learning_rate = 0.05`, `max_depth = 5`, `min_data_in_leaf = 10`)[cite: 2, 3, 5, 6, 7, 8].

### Metrics & Logged Outputs
* **Prediction accuracy**: **MSE** (Mean Squared Error) and **MAE** (Mean Absolute Error)[cite: 2, 3, 4, 5, 6, 7, 8].
* **Computational cost**: Per-run execution time (in seconds), summarized via mean, maximum, and standard deviation[cite: 3, 4, 5, 7, 8].
* **Reproducibility & checkpointing**: Simulation scripts support parallel execution (`parallel`, `pbapply`), `.rds` checkpoints, and saving seeds, train/test splits, and test predictions[cite: 3, 4, 7]. `cv_nls.R` uses a fixed subject-level 5-fold split and writes outputs to `cv_results/`[cite: 5].

---

## Simulation Setup

* **Sample size (`n`)**: `300` subjects (split 80% training / 20% testing by subject `ID`)[cite: 3, 4, 7].
* **Repeated measurements (`Time`)**: `5`, `10`, and `20`[cite: 3, 4, 7].
* **Predictors (`x1`–`x10`)**: Continuous (`x1`–`x5`), binary (`x6`–`x8`), and Poisson count (`x9`–`x10`) variables[cite: 3, 4, 7].
* **Correlation structures & `rho` settings**:
  * **AR (`sim_ar.R`)**: AR(1) correlation structure evaluated across `rho = 0.1, 0.3, 0.5`[cite: 7].
  * **UN linear (`sim_un.R`)**: Positive-definite unstructured correlation matrices generated via `make_un_corr()` across three target mean correlation levels (`low`: `0.25`, `mid`: `0.55`, `high`: `0.80`) with linear mean `X %*% beta`[cite: 2, 3].
  * **Nonlinear (`sim_un_hajjem.R`)**: Unstructured correlation matrices generated via `make_un_corr()` (`low = 0.25`, `mid = 0.55`, `high = 0.80`) with a nonlinear mean function `hajjem_F()` (depending on `x1`, `x6`, and `x9`, with remaining predictors acting as noise/distractors) scaled to unit variance[cite: 2, 4].

---

## Repository Structure

### 1. Simulation Scripts
* `function.R`: Core helper library for simulations, including `make_un_corr()`, `fit_gee()`, `fit_gpboost()`, `predict_gpboost()`, `fit_reemf()`, `pred_reemf()`, `fit_merf()`, `pred_merf()`, `.predict_longituRF_new()`, `geebm()`, `predict.geebm()`, `geebm.importance()`, `geebm.importance.plot()`, and `calculate_metrics()`[cite: 2].
* `sim_ar.R`: Parallel simulation script for AR(1) correlation (`rho = 0.1, 0.3, 0.5`) with batch checkpointing and resume support[cite: 7].
* `sim_un.R`: Parallel simulation script for linear mean with unstructured correlation[cite: 3].
* `sim_un_hajjem.R`: Parallel simulation script for nonlinear `hajjem_F` mean with unstructured correlation[cite: 4].

### 2. Real Data Analysis (`nlswork`)
* `real_function.R`: Helper functions for `nlswork` modeling and prediction (`get_x_cols()`, `fit_gee()`, `fit_gpboost()`, `predict_gpboost()`, `fit_sreemf()`, `pred_sreemf()`, `fit_smerf()`, `pred_smerf()`, `geebm()`, `predict.geebm()`, `calculate_metrics()`)[cite: 5, 6, 8].
* `sim_nls.R`: Train/test split experiment on `sampleSelection::nlswork` (`ln_wage` as outcome, complete cases, quadratic terms for `age`, `ttl_exp`, and `tenure`, plus dummy-encoded categorical variables)[cite: 8].
* `cv_nls.R`: Subject-level 5-fold cross-validation on `nlswork` across `gee` (`independence`, `ar1`, `unstructured`), `geeb` (`independence`, `ar1`, `unstructured`), `merf`, `reemforest`, and `gpboost`, saving `cv_fold_assignment.csv`, `cv_fold_level_results.csv`, `cv_summary.csv`, `cv_results.rds`, and `cv_sessionInfo.txt` under `cv_results/`[cite: 5].

### 3. Version Control
* `.gitignore`: Excludes RStudio temporary files, `.rds`/`.RData` outputs, and data files.

---

## Dependencies & Usage

### R Packages
* Modeling & statistics: `geepack`, `gpboost`, `LongituRF`, `randomForest`, `MASS`, `Matrix`[cite: 2, 3, 4, 5, 6, 7, 8].
* Data & real dataset: `sampleSelection`, `fastDummies`, `lubridate`[cite: 5, 8].
* Parallelism, plotting & paths: `parallel`, `pbapply`, `ggplot2`, `rstudioapi`[cite: 2, 3, 4, 5, 7, 8].

### Running the Scripts
1. Place all scripts in the same project directory so `function.R` and `real_function.R` can be sourced cleanly[cite: 2, 3, 4, 5, 6, 7, 8].
2. Configure `n`, `Time`, `rho`, `times`, and `n_cores` as needed before running `sim_ar.R`, `sim_un.R`, or `sim_un_hajjem.R`[cite: 3, 4, 7].
3. Run `sim_nls.R` or `cv_nls.R` for `nlswork` evaluation (`cv_nls.R` writes results to `cv_results/`)[cite: 5, 8].
