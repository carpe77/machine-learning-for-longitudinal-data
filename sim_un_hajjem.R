rm(list=ls())
library(parallel)
library(pbapply)
library(Matrix)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("function.R")


n        <- 300    # 受試者人數
Time     <- 20      # 每個人的觀測時間點
response <- "reg"   # "reg" 為回歸, "cla" 為分類
rho      <- "high"  # UN 共變異數強度: "low" (目標平均相關 0.25), "mid" (0.55), "high" (0.80)
times    <- 1000    # 重複做幾次

methods <- c("GEE", "GEEBM", "REEMF", "MERF", "GPBoost")

rho_target <- switch(rho,
                    "low"  = 0.25,
                    "mid"  = 0.55,
                    "high" = 0.80,
                    stop("rho 必須為 'low'、'mid' 或 'high'")
)


# 2. 設定 Y 的真實迴歸函數：改用 GPBoost 論文 (Sigrist, 2022) 第 4.1 節的 'hajjem' 非線性設定
#    F(x) = C * (2*x1 + x2^2 + 4*1{x3>0} + 2*log(|x1|)*x3)
#    只有 x1, x6, x9 進入真實函數，其餘變數為雜訊/干擾變數，
#    用來檢驗各方法的變數選擇能力。常數 C 使 Var(F(x)) 大致等於 1。
hajjem_F <- function(X_mat) {
  x1v <- X_mat[, 1]; x2v <- X_mat[, 6]; x3v <- X_mat[, 9]
  2 * x1v + x2v^2 + 4 * (x3v > 0) + 2 * log(abs(x1v)) * x3v
}

# ── 單次模擬函數（供平行呼叫）────────────────────────────────────────────────
run_one_sim <- function(seed, n, Time, response, rho_target, methods) {
  set.seed(seed)
  
  result <- tryCatch({
    # --- 生成資料 ---
    sigma_un <- make_un_corr(t = Time, rho_target = rho_target)
    
    # 第一階段：先產生每個受試者的預測變數矩陣 X
    X_list <- vector("list", n)
    for (i in 1:n) {
      x1_5  <- replicate(5, MASS::mvrnorm(1, mu = rep(0, Time), Sigma = 3 * sigma_un))
      z6_8  <- replicate(3, MASS::mvrnorm(1, mu = rep(0, Time), Sigma = sigma_un))
      x6_8  <- ifelse(pnorm(z6_8) > 0.5, 1, 0)
      z9_10 <- replicate(2, MASS::mvrnorm(1, mu = rep(0, Time), Sigma = sigma_un))
      x9_10 <- qpois(pnorm(z9_10), lambda = 2)
      
      X_mat <- cbind(x1_5, x6_8, x9_10)
      colnames(X_mat) <- paste0("x", 1:10)
      X_list[[i]] <- X_mat
    }
    
    # 第二階段：計算未校正的 hajjem F(x)，再用全體資料估計尺度常數 C
    F_raw_list <- lapply(X_list, hajjem_F)
    F_raw_all  <- unlist(F_raw_list)
    F_raw_all  <- F_raw_all[is.finite(F_raw_all)]
    C_hajjem   <- 1 / stats::sd(F_raw_all)
    
    # 第三階段：用校正後的 F(x) 當作平均函數，加上 UN 共變異數的殘差/機率產生 Y
    data_list <- vector("list", n)
    for (i in 1:n) {
      X_mat <- X_list[[i]]
      eta   <- C_hajjem * F_raw_list[[i]]
      
      if (response == "reg") {
        epsilon <- MASS::mvrnorm(1, mu = rep(0, Time), Sigma = 0.5 * sigma_un)
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
    all_ids    <- unique(data$ID)
    train_ids  <- sample(all_ids, round(length(all_ids) * 0.8))
    train_data <- data[ data$ID %in% train_ids, ]
    test_data  <- data[!data$ID %in% train_ids, ]
    actual_y   <- test_data$Y
    
    timings <- setNames(numeric(length(methods)), methods)
    preds   <- list()
    
    # GEE
    t0 <- Sys.time()
    m_gee     <- fit_gee(train_data, family = gaussian, corstr = "unstructured")
    preds$GEE <- as.numeric(predict(m_gee, test_data))
    timings["GEE"] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    # GEEBM
    t0 <- Sys.time()
    m_geebm     <- geebm(formula = formula_geebm, id = "ID", data = train_data,
                         iteration = 100, feature_rate = 1.0, lrate = 0.1,
                         corstr = "unstructured", family = "gaussian")
    preds$GEEBM <- as.numeric(predict.geebm(m_geebm, data = test_data))
    timings["GEEBM"] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    # REEMF
    t0 <- Sys.time()
    m_reemf     <- fit_reemf(train_data, ntree = 100)
    preds$REEMF <- pred_reemf(m_reemf, test_data)
    timings["REEMF"] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    # MERF
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
    
    # 【新增】把這次的 seed 與每個方法在測試集上的預測值一併回傳
    list(success   = TRUE,
         seed      = seed,
         mse       = mse_row, mae = mae_row, time = timings,
         preds     = preds[methods],                   # 各方法的測試集預測值（named list）
         actual_y  = as.numeric(actual_y),             # 測試集真實 Y
         test_key  = test_data[, c("ID", "Time")],     # 測試集每列對應的 ID / Time
         train_ids = sort(train_ids),                  # 這次的訓練集 ID
         C_hajjem  = C_hajjem)                         # 這次資料生成用的尺度常數
    
  }, error = function(e) {
    list(success = FALSE, seed = seed, error = conditionMessage(e))  # 【新增】失敗也記錄 seed
  })
  
  return(result)
}

# ── 平行執行（含中斷續跑機制）────────────────────────────────────────────────
n_cores = 5
# n_cores <- max(1, floor(detectCores() * 3 / 4))
cat("使用", n_cores, "個核心進行平行運算\n")

# checkpoint 檔案：存放目前為止「已成功」的結果 + 已用過的 seed 池
checkpoint_file <- sprintf("%s_n=%d_rho=%s_Time=%d.rds",
                           "nonlinear", n, rho, Time)

if (file.exists(checkpoint_file)) {
  cat("發現 checkpoint 檔案，讀取先前進度：", checkpoint_file, "\n")
  ckpt       <- readRDS(checkpoint_file)
  successful <- ckpt$successful
  seeds      <- ckpt$seeds          # 之前已經抽好、之後會繼續用同一套 seed 池
  used_idx   <- ckpt$used_idx       # 已經用到 seeds 的第幾個位置
  failed     <- ckpt$failed         # 失敗的 seed 與錯誤訊息
  if (is.null(failed)) failed <- list()
  if (length(successful) > 0 &&
      !all(vapply(successful, function(r) !is.null(r$seed) && !is.null(r$preds), logical(1)))) {
    stop("checkpoint 是舊版格式（沒有 seed/預測值），請刪除 ", checkpoint_file, " 後重跑")
  }
  cat("已累積成功次數：", length(successful), "/", times, "\n")
} else {
  successful <- list()
  failed     <- list()   # 【新增】
  seeds      <- sample.int(1e6, times * 5)  # 預先抽多一點，供重試用
  used_idx   <- 0
}

# 存檔小函式：把目前進度寫入 checkpoint_file
save_checkpoint <- function() {
  saveRDS(list(successful = successful, failed = failed, seeds = seeds, used_idx = used_idx),
          checkpoint_file)
}

if (length(successful) < times) {
  
  cl <- makeCluster(n_cores)
  
  clusterExport(cl, c(
    "run_one_sim", "make_un_corr", "hajjem_F",
    "fit_gee",
    "fit_reemf", "pred_reemf",
    "fit_merf", "pred_merf", ".predict_longituRF_new",
    "geebm", "predict.geebm", "cround",
    "fit_gpboost", "predict_gpboost",
    "n", "Time", "response", "rho_target", "methods"
  ))
  
  clusterEvalQ(cl, {
    library(MASS); library(Matrix); library(geepack)
    library(LongituRF); library(randomForest); library(gpboost)
  })
  
  # 出現非預期錯誤（例如 cluster 掛掉）時，先把目前進度存檔再往外拋錯，
  # 這樣重新執行整個 script 時可以從 checkpoint 接續，不用整批重跑
  on.exit({
    if (exists("cl")) try(stopCluster(cl), silent = TRUE)
    save_checkpoint()
    cat("\n（已將目前進度存檔至", checkpoint_file, "）\n")
  }, add = TRUE)
  
  start_time <- Sys.time()
  batch_size <- n_cores    # 每批跑幾次就存一次檔，可自行調整
  
  # 重新建立 cluster 的小函式（worker 掛掉時用來重開）
  rebuild_cluster <- function() {
    try(stopCluster(cl), silent = TRUE)
    cl <<- makeCluster(n_cores)
    clusterExport(cl, c(
      "run_one_sim", "make_un_corr", "hajjem_F",
      "fit_gee",
      "fit_reemf", "pred_reemf",
      "fit_merf", "pred_merf", ".predict_longituRF_new",
      "geebm", "predict.geebm", "cround",
      "fit_gpboost", "predict_gpboost",
      "n", "Time", "response", "rho_target", "methods"
    ))
    clusterEvalQ(cl, {
      library(MASS); library(Matrix); library(geepack)
      library(LongituRF); library(randomForest); library(gpboost)
    })
  }
  
  while (length(successful) < times) {
    
    need <- times - length(successful)
    take <- min(batch_size, need * 2)  # 多抽一些應付失敗率，避免每次都要重抽
    
    if (used_idx + take > length(seeds)) {
      seeds <- c(seeds, sample.int(1e6, take * 2))  # seed 池不夠就再補
    }
    
    batch_seeds <- seeds[(used_idx + 1):(used_idx + take)]
    
    cat("\n本批次跑", take, "次（目前已成功", length(successful), "/", times, "）...\n")
    
    # worker process 掛掉時（例如 "unserialize(node$con): 從連接讀取時發生了錯誤"），
    # pblapply 會丟出 error，這裡接住後重開 cluster 並重跑同一批 seed，
    # 而不是讓整支程式中斷、遺失進度。
    batch_results <- tryCatch({
      pblapply(batch_seeds, function(s) {
        run_one_sim(s, n, Time, response, rho_target, methods)
      }, cl = cl)
    }, error = function(e) {
      cat("\n偵測到 cluster/worker 錯誤：", conditionMessage(e), "\n")
      cat("重新建立 cluster 並重跑此批次...\n")
      rebuild_cluster()
      tryCatch({
        pblapply(batch_seeds, function(s) {
          run_one_sim(s, n, Time, response, rho_target, methods)
        }, cl = cl)
      }, error = function(e2) {
        cat("重跑後仍失敗：", conditionMessage(e2), "，這批視為全部失敗，跳過並存檔。\n")
        list()  # 這批全部視為失敗，seed 已消耗，下一輪會用新的 seed 繼續
      })
    })
    
    used_idx   <- used_idx + take   # 不論成功與否，這批 seed 都算用掉
    new_ok     <- Filter(function(r) isTRUE(r$success), batch_results)
    successful <- c(successful, new_ok)
    failed     <- c(failed, Filter(function(r) !isTRUE(r$success), batch_results))  # 【新增】記錄失敗 seed
    
    # 每跑完一批就存一次檔，中途暫停/出錯也不會遺失已完成的結果
    save_checkpoint()
  }
  
  successful <- successful[1:times]
  save_checkpoint()
  
  stopCluster(cl)
  on.exit(NULL)  # 已正常結束，取消上面 on.exit 裡重複的 stopCluster/存檔
  
  end_time <- Sys.time()
  cat("\n總執行時間：", round(difftime(end_time, start_time, units = "mins"), 2), "分鐘\n")
  
} else {
  cat("checkpoint 已達成指定次數 (", times, ")，直接使用先前結果，不再重跑。\n")
}

# ── 整理結果 ──────────────────────────────────────────────────────────────────
mse_results  <- do.call(rbind, lapply(successful, `[[`, "mse"))
mae_results  <- do.call(rbind, lapply(successful, `[[`, "mae"))
time_results <- do.call(rbind, lapply(successful, `[[`, "time"))

rownames(mse_results)  <- paste0("iter_", 1:nrow(mse_results))
rownames(mae_results)  <- paste0("iter_", 1:nrow(mae_results))
rownames(time_results) <- paste0("iter_", 1:nrow(time_results))

#把每次的 seed 與各方法預測值存成 rds ─────────────────────────────
pred_file <- sprintf("%s_pred_n=%d_rho=%s_Time=%d.rds", "nonlinear", n, rho, Time)

# 每次迭代一筆：seed、訓練 ID、測試集 key、真實值、各方法預測值、指標
runs <- lapply(successful, function(r) {
  list(seed = r$seed, C_hajjem = r$C_hajjem, train_ids = r$train_ids,
       test_key = r$test_key, actual_y = r$actual_y, preds = r$preds,
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
                      rho_target = rho_target, times = times, methods = methods),
  seeds        = vapply(successful, function(r) as.integer(r$seed), integer(1)),  # 成功的每次 seed
  failed_seeds = vapply(failed, function(r) as.integer(r$seed), integer(1)),      # 失敗的 seed
  seed_pool    = seeds[seq_len(used_idx)],                                        # 全部用過的 seed
  runs         = runs,
  pred_table   = pred_table
), pred_file)
cat("\n已儲存 seed 與預測值至：", pred_file, "\n")

mse_avg <- colMeans(mse_results, na.rm = TRUE)
mae_avg <- colMeans(mae_results, na.rm = TRUE)

cat("\n── 每次迭代 MSE ──\n")
print(round(mse_results, 6))
cat("\n── 每次迭代 MAE ──\n")
print(round(mae_results, 6))

cat("\n── MSE ──\n")
print(format(round(mse_avg, 6), nsmall = 6), quote = FALSE)
cat("\n── MAE ──\n")
print(format(round(mae_avg, 6), nsmall = 6), quote = FALSE)
cat("\n── 平均執行時間（秒）──\n")
print(colMeans(time_results))
cat("\n── 最長執行時間（秒）──\n")
for (i in seq_along(methods)) {
  cat(methods[i], ": max =", round(max(time_results[, i]), 2),
      ", sd =", round(sd(time_results[, i]), 2), "\n")
}

# ── 極端值檢查 ────────────────────────────────────────────────────────────────
cat("\n── 各方法 MSE 超過 100 的次數 ──\n")
for (m in methods) {
  cat(m, ":", sum(mse_results[, m] > 100), "次\n")
}
cat("\n── 各方法 MSE 最大的前 5 次 ──\n")
for (m in methods) {
  cat(m, ":\n")
  print(sort(mse_results[, m], decreasing = TRUE)[1:5])
}
for(i in 1:length(methods)){
  cat(methods[i],"mse:",sd(mse_results[,i]))
  cat(methods[i],"mae:",sd(mae_results[,i]))
  cat("\n")
}
