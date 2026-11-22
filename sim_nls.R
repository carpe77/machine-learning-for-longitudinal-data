rm(list = ls())
library(lubridate)
library(sampleSelection)   # 用來載入 nlswork
library(fastDummies)       # dummy_cols()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("real_function.R")

# ── 0. 選擇要跑的方法 ─────────────────────────────────────────────────
methods <- c("GEE", "GEEBM", "REEMF", "MERF", "GPBoost")


# GEE / GEEBM 的 working correlation："independence"、"ar1"、"unstructured"
corstr_use  <- "independence"

cat("本次執行的方法：", paste(methods, collapse = ", "), "\n")


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


# ── 2. 各方法的 fit + predict（回傳 test 預測值）─────────────────────
run_funs <- list(
  GEE = function(tr, te) {
    m <- fit_gee(tr, family = gaussian, corstr = corstr_use)
    predict(m, te)
  },
  GEEBM = function(tr, te) {
    m <- geebm(
      formula      = formula_all,
      id           = "ID",
      data         = tr,
      iteration    = 100,
      feature_rate = 1.0,
      lrate        = 0.1,
      corstr       = corstr_use,
      family       = "gaussian"
    )
    predict.geebm(m, data = te)
  },
  # REEMforest, sto = "none", ntree = 100, iter = 50, delta = 0.01
  REEMF = function(tr, te) {
    m <- fit_reemf(tr, ntree = 100)
    pred_reemf(m, te)
  },
  # MERF, sto = "none", ntree = 100
  MERF = function(tr, te) {
    m <- fit_merf(tr, ntree = 100)
    pred_merf(m, te)
  },
  GPBoost = function(tr, te) {
    m <- fit_gpboost(tr, response_type = "reg")
    predict_gpboost(m, te)
  }
)

# ── 3. 模擬設定（只保留重複次數）────────────────────────────────────
times     <- 1
base_seed <- 50   # 每次 split 使用 set.seed(base_seed + j)，確保可重現

mse_results  <- matrix(NA, nrow = times, ncol = length(methods),
                       dimnames = list(NULL, methods))
mae_results  <- matrix(NA, nrow = times, ncol = length(methods),
                       dimnames = list(NULL, methods))
time_results <- matrix(NA, nrow = times, ncol = length(methods),
                       dimnames = list(NULL, methods))

# ── 4. 重複模擬（只有 train/test split 是隨機的）────────────────────
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
  
  # 只跑 methods 中選取的方法；某方法失敗時記為 NA，不中斷其他方法
  for (m in methods) {
    cat("  執行", m, "...\n")
    start_t <- Sys.time()
    pred <- tryCatch(
      as.numeric(run_funs[[m]](train_data, test_data)),
      error = function(e) {
        message("  ", m, " 失敗：", conditionMessage(e))
        rep(NA_real_, nrow(test_data))
      }
    )
    time_results[j, m] <- as.numeric(difftime(Sys.time(), start_t, units = "secs"))
    
    # 計算 MSE / MAE（只算選取的方法）
    met <- calculate_metrics(actual_y, pred)
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
