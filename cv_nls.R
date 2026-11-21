###############################################################################
# Subject-level 5-fold cross-validation on NLSY (nlswork)
#
# - Folds are built on subject ID (all observations of a subject stay together)
# - Folds are generated ONCE and shared by every method
# - Methods (9):
#     GEE_IND, GEE_AR1, GEE_UN        (GEE with 3 working correlations)
#     GEEB_IND, GEEB_AR1, GEEB_UN     (GEEB with 3 working correlations)
#     MERF, REEMforest, GPBoost       (one fit per fold)
# - Hyperparameters = prespecified settings in real_function.R (no nested CV)
# - Saved per fold & method: #train/#test subjects, #test obs, MSE, MAE,
#   runtime, success/failure (+ error / warning message)
###############################################################################

rm(list = ls())
library(lubridate)
library(sampleSelection)   # nlswork data
library(fastDummies)       # dummy_cols()

# ── 0. Working directory & helper functions ────────────────────────────────
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}
function_file <- "real function.R"                       # 本機檔名（含空格）
if (!file.exists(function_file)) function_file <- "real_function.R"
source(function_file)    # get_x_cols, fit_*, pred_*, geebm, predict.geebm, calculate_metrics

# ── 1. CV settings ─────────────────────────────────────────────────────────
K         <- 5
cv_seed   <- 2026     # 只用來決定 folds（固定一次）
base_seed <- 123      # 每個 fold × method 的模型隨機種子
out_dir   <- "cv_results"
dir.create(out_dir, showWarnings = FALSE)

# ── 2. Load & clean nlswork（與 sim_nls.R 完全相同）────────────────────────
data(nlswork)

target <- "ln_wage"
base_features <- c("age", "ttl_exp", "tenure", "not_smsa", "south", "year",
                   "msp", "nev_mar", "collgrad", "c_city", "hours", "grade")
cat_features  <- c("ind_code", "occ_code", "race")

use_cols <- c("idcode", target, base_features, cat_features)
nlswork_clean <- nlswork[, use_cols]
nlswork_clean <- na.omit(nlswork_clean)

nlswork_clean$age_sq     <- nlswork_clean$age^2
nlswork_clean$ttl_exp_sq <- nlswork_clean$ttl_exp^2
nlswork_clean$tenure_sq  <- nlswork_clean$tenure^2

nlswork_clean <- dummy_cols(
  nlswork_clean,
  select_columns = cat_features,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

nlswork_clean <- nlswork_clean[order(nlswork_clean$idcode, nlswork_clean$year), ]
colnames(nlswork_clean)[colnames(nlswork_clean) == "idcode"]  <- "ID"
colnames(nlswork_clean)[colnames(nlswork_clean) == "ln_wage"] <- "Y"

# Time：只給 MERF / REEMforest 的 time 引數用，不是 predictor
nlswork_clean$Time <- nlswork_clean$year - min(nlswork_clean$year)

data   <- nlswork_clean
x_vars <- get_x_cols(data)                      # 排除 ID、Y、Time
p      <- length(x_vars)
formula_all <- as.formula(paste("Y ~", paste(x_vars, collapse = " + ")))

cat("Subjects:", length(unique(data$ID)), "| Observations:", nrow(data),
    "| Predictors:", p, "\n")

# ── 3. Subject-level folds（固定一次，所有方法共用）─────────────────────────
set.seed(cv_seed)
all_ids  <- unique(data$ID)
fold_of  <- sample(rep(1:K, length.out = length(all_ids)))   # 各 fold 人數盡量相等
fold_map <- data.frame(ID = all_ids, fold = fold_of)
data$fold <- fold_map$fold[match(data$ID, fold_map$ID)]

write.csv(fold_map, file.path(out_dir, "cv_fold_assignment.csv"), row.names = FALSE)

# ── 4. Method specifications ───────────────────────────────────────────────
# GEE / GEEB 的 working correlation：independence / ar1 / unstructured
# （未指定 waves：沿用 geepack 預設，以各 subject 內的觀測順序作為 wave，
#   與模擬程式一致）
cor_map <- c(IND = "independence", AR1 = "ar1", UN = "unstructured")

method_specs <- list()
for (cc in names(cor_map)) {
  local({
    cs <- cor_map[[cc]]
    method_specs[[paste0("GEE_", cc)]] <<- list(
      fit  = function(d) fit_gee(d, family = gaussian, corstr = cs),
      pred = function(m, d) predict(m, d)
    )
    method_specs[[paste0("GEEB_", cc)]] <<- list(
      fit  = function(d) geebm(formula = formula_all, id = "ID", data = d,
                               iteration = 100, feature_rate = 1.0, lrate = 0.1,
                               corstr = cs, family = "gaussian"),
      pred = function(m, d) predict.geebm(m, data = d)
    )
  })
}
method_specs[["MERF"]] <- list(
  fit  = function(d) fit_smerf(d, ntree = 100),                       # sto = "none"
  pred = function(m, d) pred_smerf(m, d)
)
method_specs[["REEMforest"]] <- list(
  fit  = function(d) fit_sreemf(d, ntree = 100, iter = 50, delta = 0.01),  # sto = "none"
  pred = function(m, d) pred_sreemf(m, d)
)
method_specs[["GPBoost"]] <- list(
  fit  = function(d) fit_gpboost(d, response_type = "reg"),
  pred = function(m, d) predict_gpboost(m, d)
)

method_names <- c("GEE_IND", "GEE_AR1", "GEE_UN",
                  "GEEB_IND", "GEEB_AR1", "GEEB_UN",
                  "MERF", "REEMforest", "GPBoost")
method_specs <- method_specs[method_names]

# ── 5. Run one method on one fold (never stops the loop) ───────────────────
run_method <- function(spec, train, test) {
  warn_msgs <- character(0)
  t0 <- Sys.time()
  res <- tryCatch(
    withCallingHandlers({
      fit  <- spec$fit(train)
      pred <- as.numeric(spec$pred(fit, test))
      if (length(pred) != nrow(test)) stop("prediction length != number of test observations")
      if (any(!is.finite(pred)))      stop("non-finite predictions (NA/NaN/Inf)")
      list(ok = TRUE, pred = pred, msg = "")
    }, warning = function(w) {
      warn_msgs <<- c(warn_msgs, conditionMessage(w))
      invokeRestart("muffleWarning")
    }),
    error = function(e) list(ok = FALSE, pred = NULL, msg = conditionMessage(e))
  )
  runtime <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  if (res$ok) {
    met <- calculate_metrics(test$Y, res$pred)
    mse <- unname(met["MSE"]); mae <- unname(met["MAE"])
  } else {
    mse <- NA_real_; mae <- NA_real_
  }
  list(MSE = mse, MAE = mae, runtime_sec = runtime,
       status = ifelse(res$ok, "success", "failure"),
       error_message = res$msg,
       warning_message = paste(unique(warn_msgs), collapse = " | "))
}

# ── 6. Cross-validation loop ───────────────────────────────────────────────
rows <- list()
total_start <- Sys.time()

for (k in 1:K) {
  train_data <- data[data$fold != k, ]
  test_data  <- data[data$fold == k, ]
  
  # 檢查：subject-level split（train/test 無重疊 subject）
  stopifnot(length(intersect(train_data$ID, test_data$ID)) == 0)
  
  n_train_subj <- length(unique(train_data$ID))
  n_test_subj  <- length(unique(test_data$ID))
  cat(sprintf("\n=== Fold %d/%d | train subj = %d | test subj = %d | test obs = %d ===\n",
              k, K, n_train_subj, n_test_subj, nrow(test_data)))
  
  for (mi in seq_along(method_names)) {
    m <- method_names[mi]
    set.seed(base_seed + k * 100 + mi)
    r <- run_method(method_specs[[m]], train_data, test_data)
    
    rows[[length(rows) + 1]] <- data.frame(
      fold            = k,
      method          = m,
      n_train_subj    = n_train_subj,
      n_test_subj     = n_test_subj,
      n_train_obs     = nrow(train_data),
      n_test_obs      = nrow(test_data),
      MSE             = r$MSE,
      MAE             = r$MAE,
      runtime_sec     = r$runtime_sec,
      status          = r$status,
      error_message   = r$error_message,
      warning_message = r$warning_message,
      stringsAsFactors = FALSE
    )
    cat(sprintf("  %-11s %-8s MSE = %s | MAE = %s | %.1f s %s\n",
                m, r$status,
                ifelse(is.na(r$MSE), "NA", format(round(r$MSE, 5), nsmall = 5)),
                ifelse(is.na(r$MAE), "NA", format(round(r$MAE, 5), nsmall = 5)),
                r$runtime_sec,
                ifelse(r$status == "failure", paste0("<- ", r$error_message), "")))
  }
}
cat("\nTotal time:", round(difftime(Sys.time(), total_start, units = "mins"), 2), "min\n")

fold_results <- do.call(rbind, rows)
fold_results$method <- factor(fold_results$method, levels = method_names)
fold_results <- fold_results[order(fold_results$method, fold_results$fold), ]
fold_results$method <- as.character(fold_results$method)

# ── 7. Summary: mean ± SD across folds (successful folds only) ─────────────
summary_tab <- do.call(rbind, lapply(method_names, function(m) {
  d  <- fold_results[fold_results$method == m, ]
  ok <- d[d$status == "success", ]
  n_ok <- nrow(ok)
  data.frame(
    method           = m,
    n_folds_success  = n_ok,
    n_folds_failed   = nrow(d) - n_ok,
    mean_MSE         = if (n_ok > 0) mean(ok$MSE) else NA_real_,
    sd_MSE           = if (n_ok > 1) sd(ok$MSE)   else NA_real_,
    mean_MAE         = if (n_ok > 0) mean(ok$MAE) else NA_real_,
    sd_MAE           = if (n_ok > 1) sd(ok$MAE)   else NA_real_,
    mean_runtime_sec = mean(d$runtime_sec),
    stringsAsFactors = FALSE
  )
}))
summary_tab$MSE_mean_pm_SD <- ifelse(is.na(summary_tab$mean_MSE), "NA",
                                     sprintf("%.4f \u00b1 %.4f", summary_tab$mean_MSE, summary_tab$sd_MSE))
summary_tab$MAE_mean_pm_SD <- ifelse(is.na(summary_tab$mean_MAE), "NA",
                                     sprintf("%.4f \u00b1 %.4f", summary_tab$mean_MAE, summary_tab$sd_MAE))

# ── 8. Print & save ────────────────────────────────────────────────────────
cat("\n===== Summary (mean \u00b1 SD over successful folds) =====\n")
print(summary_tab[, c("method", "n_folds_success", "n_folds_failed",
                      "MSE_mean_pm_SD", "MAE_mean_pm_SD", "mean_runtime_sec")],
      row.names = FALSE)

cat("\n===== Fold-level MSE =====\n")
print(round(tapply(fold_results$MSE, list(fold_results$fold, fold_results$method), identity)[, method_names], 5))
cat("\n===== Fold-level MAE =====\n")
print(round(tapply(fold_results$MAE, list(fold_results$fold, fold_results$method), identity)[, method_names], 5))

write.csv(fold_results, file.path(out_dir, "cv_fold_level_results.csv"), row.names = FALSE)
write.csv(summary_tab,  file.path(out_dir, "cv_summary.csv"),            row.names = FALSE)
saveRDS(list(fold_results = fold_results, summary = summary_tab,
             fold_map = fold_map, settings = list(K = K, cv_seed = cv_seed, base_seed = base_seed)),
        file.path(out_dir, "cv_results.rds"))
capture.output(sessionInfo(), file = file.path(out_dir, "cv_sessionInfo.txt"))
cat("\nResults saved in:", normalizePath(out_dir), "\n")
