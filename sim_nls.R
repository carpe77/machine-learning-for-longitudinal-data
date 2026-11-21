rm(list = ls())
library(lubridate)
library(sampleSelection)   # 用來載入 nlswork
library(fastDummies)       # dummy_cols()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("real_function.R")

# ── 1. 載入並清理 nlswork ────────────────────────────────────────────
data(nlswork)

# response variable
target <- "ln_wage"

# 原始 predictor variables
base_features <- c(
  "age",
  "ttl_exp",
  "tenure",
  "not_smsa",
  "south",
  "year",
  "msp",
  "nev_mar",
  "collgrad",
  "c_city",
  "hours",
  "grade"
)

# low-cardinality categorical variables
cat_features <- c(
  "ind_code",
  "occ_code",
  "race"
)

# 保留欄位
use_cols <- c(
  "idcode",
  target,
  base_features,
  cat_features
)

nlswork_clean <- nlswork[, use_cols]

# 移除缺失值
nlswork_clean <- na.omit(nlswork_clean)

# ─────────────────────────────────────────────────────
# 2. 建立平方項
# ─────────────────────────────────────────────────────
nlswork_clean$age_sq     <- nlswork_clean$age^2
nlswork_clean$ttl_exp_sq <- nlswork_clean$ttl_exp^2
nlswork_clean$tenure_sq  <- nlswork_clean$tenure^2

# ─────────────────────────────────────────────────────
# 3. Dummy coding
# ─────────────────────────────────────────────────────
nlswork_clean <- dummy_cols(
  nlswork_clean,
  select_columns = cat_features,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

# ─────────────────────────────────────────────────────
# 4. 排序
# ─────────────────────────────────────────────────────
nlswork_clean <- nlswork_clean[
  order(nlswork_clean$idcode, nlswork_clean$year),
]

# ─────────────────────────────────────────────────────
# 5. 統一欄位名稱
# ─────────────────────────────────────────────────────
colnames(nlswork_clean)[
  colnames(nlswork_clean) == "idcode"
] <- "ID"

colnames(nlswork_clean)[
  colnames(nlswork_clean) == "ln_wage"
] <- "Y"

# ─────────────────────────────────────────────────────
# 6. 建立 Time（僅作為 MERF / REEMforest 的 time 引數，不是 predictor）
# ─────────────────────────────────────────────────────
nlswork_clean$Time <- nlswork_clean$year - min(nlswork_clean$year)

data <- nlswork_clean

# predictor 清單（get_x_cols 會排除 ID、Y、Time）
x_vars <- get_x_cols(data)
x_vars

# predictor 數量
p <- length(x_vars)

# ─────────────────────────────────────────────────────
# 7. 建立自動公式
# ─────────────────────────────────────────────────────
formula_all <- as.formula(
  paste("Y ~", paste(x_vars, collapse = " + "))
)

# ── 2. 模擬設定（只保留重複次數）────────────────────────────────────
times   <- 1
base_seed <- 50   # 每次 split 使用 set.seed(base_seed + j)，確保可重現
methods <- c("GEE", "GEEBM", "SMERF", "GPBoost")

mse_results  <- matrix(NA, nrow = times, ncol = length(methods),
                       dimnames = list(NULL, methods))
mae_results  <- matrix(NA, nrow = times, ncol = length(methods),
                       dimnames = list(NULL, methods))
time_results <- matrix(NA, nrow = times, ncol = length(methods),
                       dimnames = list(NULL, methods))

# ── 3. 重複模擬（只有 train/test split 是隨機的）────────────────────
start_time <- Sys.time()

for (j in 1:times) {
  
  set.seed(base_seed + j)
  all_ids  <- unique(data$ID)
  n_total  <- length(all_ids)
  n_train  <- round(n_total * 0.8)
  train_ids <- sample(all_ids, n_train)
  
  train_data <- data[data$ID %in%  train_ids, ]
  test_data  <- data[!data$ID %in% train_ids, ]
  actual_y   <- test_data$Y
  
  # GEE
  start_t <- Sys.time()
  train_gee <- fit_gee(train_data, family = gaussian, corstr = "independence")
  test_gee  <- predict(train_gee, test_data)
  time_results[j, "GEE"] <- as.numeric(difftime(Sys.time(), start_t, units = "secs"))
  
  # # SREEMF (REEMforest, sto = "none", ntree = 100, iter = 50, delta = 0.01)
  # start_t <- Sys.time()
  # train_sreemf <- fit_sreemf(train_data, ntree = 100)
  # test_sreemf  <- pred_sreemf(train_sreemf, test_data)
  # time_results[j, "SREEMF"] <- as.numeric(difftime(Sys.time(), start_t, units = "secs"))
  
  # SMERF (MERF, sto = "none", ntree = 100)
  start_t <- Sys.time()
  train_smerf <- fit_smerf(train_data, ntree = 100)
  test_smerf  <- pred_smerf(train_smerf, test_data)
  time_results[j, "SMERF"] <- as.numeric(difftime(Sys.time(), start_t, units = "secs"))
  
  # GEEBM
  start_t <- Sys.time()
  train_geebm <- geebm(
    formula      = formula_all,
    id           = "ID",
    data         = train_data,
    iteration    = 100,
    feature_rate = 1.0,
    lrate        = 0.1,
    corstr       = "independence",
    family       = "gaussian"
  )
  test_geebm <- predict.geebm(train_geebm, data = test_data)
  time_results[j, "GEEBM"] <- as.numeric(difftime(Sys.time(), start_t, units = "secs"))
  
  # GPBoost
  start_t <- Sys.time()
  train_gpboost <- fit_gpboost(train_data, response_type = "reg")
  test_gpboost  <- predict_gpboost(train_gpboost, test_data)
  time_results[j, "GPBoost"] <- as.numeric(difftime(Sys.time(), start_t, units = "secs"))
  
  # 計算 MSE / MAE
  # all_preds <- list(GEE = test_gee, GEEBM = test_geebm,
  #                   SREEMF = test_sreemf, SMERF = test_smerf, GPBoost = test_gpboost)
  #nosreemf
  all_preds <- list(GEE = test_gee, GEEBM = test_geebm,
                     SMERF = test_smerf, GPBoost = test_gpboost)
  for (m in methods) {
    met <- calculate_metrics(actual_y, all_preds[[m]])
    mse_results[j, m] <- met["MSE"]
    mae_results[j, m] <- met["MAE"]
  }
  cat("目前運行", j, "次\n")
}

end_time <- Sys.time()
cat("總執行時間：", difftime(end_time, start_time, units = "mins"), "分鐘\n")

mse_avg <- colMeans(mse_results, na.rm = TRUE)
mae_avg <- colMeans(mae_results, na.rm = TRUE)
print(format(round(mse_avg, 6), nsmall = 6), quote = FALSE)
print(format(round(mae_avg, 6), nsmall = 6), quote = FALSE)
print(time_results)