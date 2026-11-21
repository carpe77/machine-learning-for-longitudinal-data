rm(list=ls())
library(parallel)
library(pbapply)
library(Matrix)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("function.R")   

n <- 300       # 受試者人數
Time <- 5     # 每個人的觀測時間點
response <- "reg" # "reg" 為回歸, "cla" 為分類
rho <- 0.3     # AR(1) 相關係數
times <- 1000   #重複做幾次

methods <- c("GEE", "GEEBM", "REEMF", "MERF", "GPBoost")


# 1. 建立 AR(1) 相關結構矩陣
# 這會生成一個 Time x Time 的矩陣，體現時間點越近相關性越高的特性
sigma_ar1 <- rho^abs(matrix(1:Time, Time, Time, byrow = TRUE) - (1:Time))

# 2. 設定 Y 的真實迴歸係數
beta <- c(0.8, -0.5, 0, 0, 0, 0.6, 0, 0, 0.4, 0.5)

# ── 單次模擬函數（供平行呼叫）────────────────────────────────────────────────
run_one_sim <- function(seed, n, Time, response, beta, methods) {
  set.seed(seed)
  
  result <- tryCatch({
    # --- 生成資料 ---
    data_list <- vector("list", n)
    
    for (i in 1:n) {
      x1_5  <- replicate(5, MASS::mvrnorm(1, mu = rep(0, Time), Sigma = 3 * sigma_ar1))
      z6_8  <- replicate(3, MASS::mvrnorm(1, mu = rep(0, Time), Sigma = sigma_ar1))
      x6_8  <- ifelse(pnorm(z6_8) > 0.5, 1, 0)
      z9_10 <- replicate(2, MASS::mvrnorm(1, mu = rep(0, Time), Sigma = sigma_ar1))
      x9_10 <- qpois(pnorm(z9_10), lambda = 2)
      
      X_mat <- cbind(x1_5, x6_8, x9_10)
      colnames(X_mat) <- paste0("x", 1:10)
      eta   <- X_mat %*% beta
      
      if (response == "reg") {
        epsilon <- MASS::mvrnorm(1, mu = rep(0, Time), Sigma = 0.5 * sigma_ar1)
        y <- eta + epsilon
      } else {
        prob <- 1 / (1 + exp(-eta))
        y    <- rbinom(Time, 1, prob)
      }
      
      data_list[[i]] <- data.frame(ID = i, Time = 1:Time, Y = as.vector(y), X_mat)
    }
    
    data <- do.call(rbind, data_list)
    formula_geebm <- Y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10
    
    # --- 切分訓練/測試集 ---
    all_ids   <- unique(data$ID)
    train_ids <- sample(all_ids, round(length(all_ids) * 0.8))
    train_data <- data[ data$ID %in% train_ids, ]
    test_data  <- data[!data$ID %in% train_ids, ]
    actual_y   <- test_data$Y
    y_range    <- range(train_data$Y)
    
    timings <- setNames(numeric(length(methods)), methods)
    preds   <- list()
    
    # GEE
    t0 <- Sys.time()
    m_gee     <- fit_gee(train_data, family = gaussian, corstr = "ar1")
    preds$GEE <- as.numeric(predict(m_gee, test_data))
    timings["GEE"] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    # GEEBM
    t0 <- Sys.time()
    m_geebm     <- geebm(formula = formula_geebm, id = "ID", data = train_data,
                         iteration = 100, feature_rate = 1.0, lrate = 0.1,
                         corstr = "ar1", family = "gaussian")
    preds$GEEBM <- as.numeric(predict.geebm(m_geebm, data = test_data))
    timings["GEEBM"] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    # REEMF
    t0 <- Sys.time()
    m_reemf     <- fit_reemf(train_data, ntree = 100)
    preds$REEMF <- pred_reemf(m_reemf, test_data)
    timings["REEMF"] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    # # MERF
    t0 <- Sys.time()
    m_merf     <- fit_merf(train_data, ntree = 100)
    preds$MERF <- pred_merf(m_merf, test_data)
    timings["MERF"] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    # GPBoost
    t0 <- Sys.time()
    m_gpb         <- fit_gpboost(train_data, response_type = "reg")
    preds$GPBoost <- predict_gpboost(m_gpb, test_data)
    timings["GPBoost"] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    # --- 計算指標 ---
    mse_row <- setNames(numeric(length(methods)), methods)
    mae_row <- setNames(numeric(length(methods)), methods)
    for (m in methods) {
      mse_row[m] <- mean((as.numeric(actual_y) - as.numeric(preds[[m]]))^2)
      mae_row[m] <- mean(abs(as.numeric(actual_y) - as.numeric(preds[[m]])))
    }
    
    # 把這次的 seed 與每個方法在測試集上的預測值一併回傳
    list(success   = TRUE,
         seed      = seed,
         mse       = mse_row, mae = mae_row, time = timings,
         preds     = preds[methods],                   # 各方法的測試集預測值（named list）
         actual_y  = as.numeric(actual_y),             # 測試集真實 Y
         test_key  = test_data[, c("ID", "Time")],     # 測試集每列對應的 ID / Time
         train_ids = sort(train_ids))                  # 這次的訓練集 ID
    
  }, error = function(e) {
    list(success = FALSE, seed = seed, error = conditionMessage(e))  # 失敗也記錄 seed
  })
  
  return(result)
}

# ── 平行執行（小批次進行，隨時可見目前完成次數，可中斷）───────────────────────
#
# 【提醒】核心數請依實測記憶體用量調整，不要無腦全開，可參考之前的討論方式：
#   gc(reset = TRUE); invisible(run_one_sim(1, n, Time, response, beta, methods)); gc()
#
n_cores <- 4
cat("使用", n_cores, "個核心進行平行運算（可用核心共", detectCores(), "個）\n")

cl <- parallel::makeCluster(n_cores)

# 把需要的套件與函數匯出到每個 worker
parallel::clusterExport(cl, c(
  "run_one_sim",
  "fit_gee", 
  "fit_reemf", "pred_reemf",
  "fit_merf", "pred_merf", ".predict_longituRF_new",
  "geebm", "predict.geebm", "cround",
  "fit_gpboost", "predict_gpboost",
  "n", "Time", "response", "sigma_ar1", "beta", "methods"
))
parallel::clusterEvalQ(cl, {
  library(MASS); library(Matrix); library(geepack)
  library(LongituRF); library(randomForest); library(gpboost)
})

# checkpoint 檔名依這次的設定自動產生，換設定就自動變成新的一組，不會互相污染
config_tag <- paste(
  "rho", rho, "n", n, "Time", Time, paste(methods, collapse = "-"),
  sep = "_"
)
checkpoint_file      <- paste0("sim_checkpoint_ar_", config_tag, ".rds")
partial_result_file  <- paste0("partial_results_ar_", config_tag, ".rds")
cat("本次設定標籤：", config_tag, "\n")
cat("checkpoint 檔案：", checkpoint_file, "\n")

# checkpointlist(successful, failed)，才能一併保留失敗的 seed
successful <- list()
failed     <- list()
if (file.exists(checkpoint_file)) {
  ckpt <- readRDS(checkpoint_file)
  if (length(ckpt) > 0 && is.null(ckpt$successful)) {
    stop("checkpoint 是舊版格式（沒有 seed/預測值），請刪除 ", checkpoint_file, " 後重跑")
  }
  if (length(ckpt) > 0) {
    successful <- ckpt$successful
    failed     <- ckpt$failed
    if (is.null(failed)) failed <- list()
  }
  cat("偵測到先前的中繼結果，已載入", length(successful), "筆，將接續執行\n")
}

# 每一批同時送 n_cores 個任務，跑完立刻存檔，中斷時最多只損失當下這一批
batch_size <- n_cores
seeds_pool <- sample.int(1e6, times * 3)
seed_idx   <- 1
batch_num  <- 0

save_checkpoint <- function() {
  saveRDS(list(successful = successful, failed = failed), checkpoint_file)  # 【修改】多存 failed
}

report_results <- function() {
  n_done <- length(successful)
  cat("\n========================================\n")
  cat("目前已完成模擬次數：", n_done, "/", times, "\n")
  cat("========================================\n")

  if (n_done == 0) {
    cat("目前沒有任何成功結果可計算，請再等等或檢查錯誤\n")
    return(invisible(NULL))
  }

  mse_results  <- do.call(rbind, lapply(successful, `[[`, "mse"))
  mae_results  <- do.call(rbind, lapply(successful, `[[`, "mae"))
  time_results <- do.call(rbind, lapply(successful, `[[`, "time"))

  mse_avg <- colMeans(mse_results, na.rm = TRUE)
  mae_avg <- colMeans(mae_results, na.rm = TRUE)

  cat("\n── MSE（基於", n_done, "次）──\n")
  print(format(round(mse_avg, 6), nsmall = 6), quote = FALSE)
  cat("\n── MAE（基於", n_done, "次）──\n")
  print(format(round(mae_avg, 6), nsmall = 6), quote = FALSE)
  cat("\n── 平均執行時間（秒）──\n")
  print(colMeans(time_results))
  cat("\n── 最長執行時間（秒）──\n")
  for (i in seq_along(methods)) {
    cat(methods[i], ": max =", round(max(time_results[, i]), 2),
        ", sd =", round(sd(time_results[, i]), 2), "\n")
  }

  saveRDS(list(mse = mse_results, mae = mae_results, time = time_results, n_done = n_done),
          partial_result_file)
  cat("\n(詳細結果已另存為", partial_result_file, "，共", n_done, "筆)\n")

  # ── 【新增】把每次的 seed 與各方法預測值存成 rds（中斷時也會存目前進度）────────
  pred_file <- paste0("predictions_ar_", config_tag, ".rds")

  # 每次迭代一筆：seed、訓練 ID、測試集 key、真實值、各方法預測值、指標
  runs <- lapply(successful, function(r) {
    list(seed = r$seed, train_ids = r$train_ids, test_key = r$test_key,
         actual_y = r$actual_y, preds = r$preds,
         mse = r$mse, mae = r$mae, time = r$time)
  })
  names(runs) <- paste0("iter_", seq_along(runs))

  # 攤平成一張表：iter, seed, ID, Time, actual, 各方法預測值（方便直接分析/畫圖）
  pred_table <- do.call(rbind, lapply(seq_along(successful), function(i) {
    r <- successful[[i]]
    data.frame(iter = i, seed = r$seed, r$test_key, actual = r$actual_y,
               as.data.frame(r$preds), row.names = NULL)
  }))

  saveRDS(list(
    settings     = list(n = n, Time = Time, response = response, rho = rho,
                        times = times, n_done = n_done, methods = methods, beta = beta),
    seeds        = vapply(successful, function(r) as.integer(r$seed), integer(1)),  # 成功的每次 seed
    failed_seeds = vapply(failed, function(r) as.integer(r$seed), integer(1)),      # 失敗的 seed
    runs         = runs,
    pred_table   = pred_table
  ), pred_file)
  cat("已儲存 seed 與預測值至：", pred_file, "\n")

  assign("mse_results", mse_results, envir = .GlobalEnv)
  assign("mae_results", mae_results, envir = .GlobalEnv)
  assign("time_results", time_results, envir = .GlobalEnv)
}

start_time <- Sys.time()

run_status <- tryCatch({

  while (length(successful) < times) {
    need <- min(batch_size, times - length(successful))
    batch_num <- batch_num + 1

    if (seed_idx + need - 1 > length(seeds_pool)) {
      seeds_pool <- c(seeds_pool, sample.int(1e6, times * 2))
    }
    batch_seeds <- seeds_pool[seed_idx:(seed_idx + need - 1)]
    seed_idx <- seed_idx + need

    cat("\n第", batch_num, "批次執行中（本批", need, "筆，累計完成",
        length(successful), "/", times, "）...\n")

    batch_results <- pblapply(batch_seeds, function(s) {
      run_one_sim(s, n, Time, response, beta, methods)
    }, cl = cl)

    ok         <- Filter(function(r) isTRUE(r$success), batch_results)
    successful <<- c(successful, ok)
    failed     <<- c(failed, Filter(function(r) !isTRUE(r$success), batch_results))  # 【新增】記錄失敗 seed

    save_checkpoint()
    cat("本批次成功", length(ok), "/", need,
        "，累計完成", length(successful), "/", times, "\n")
  }
  "completed"

}, interrupt = function(e) {
  cat("\n\n⚠ 偵測到中斷指令（Stop / Ctrl+C），停止繼續送出新任務。\n")
  cat("正在使用目前已完成的", length(successful), "筆結果計算...\n")
  "interrupted"
})

stopCluster(cl)

end_time <- Sys.time()
cat("\n總執行時間：", round(difftime(end_time, start_time, units = "mins"), 2), "分鐘\n")
cat("執行狀態：", run_status, "（completed = 全部跑完；interrupted = 中途被中斷）\n")

report_results()

for(i in 1:length(methods)){
  cat(methods[i],"mse:",sd(mse_results[,i]))
  cat(methods[i],"mae:",sd(mae_results[,i]))
  cat("\n")
}
