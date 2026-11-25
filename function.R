library(MASS)
library(Matrix)
library(gpboost)
library(LongituRF)
library(geepack)
library(gpboost)
library(ggplot2)
logistic <- function(x){ (1+exp(-x))^(-1) }

#' 生成 Unstructured (UN) 正定相關矩陣，直接控制「平均相關係數」
#'
#' @param t 時間點個數
#' @param rho_target 目標平均 off-diagonal 相關係數 (例如 low=0.25, mid=0.55, high=0.80)
#' @param df 用來生成隨機基底相關矩陣 C 的自由度。df 越大，C 的 off-diagonal
#'   相關越集中在 0 附近（變異數越小），愈容易在低 rho_target（如 0.25）時
#' @param positive_only 是否強制要求所有 pairwise correlation 皆 > 0（預設 TRUE，
#'   符合大多數 UN 共變異數模擬情境的假設）
#' @param max_tries 最多嘗試次數，避免無窮迴圈
#' @return 一個 t x t 的正定相關矩陣 R，其平均 off-diagonal 元素等於 rho_target
make_un_corr <- function(t,
                         rho_target,
                         df = 100,
                         positive_only = TRUE,
                         max_tries = 1000) {
  
  stopifnot(t >= 2)
  stopifnot(rho_target > 0 && rho_target < 1)
  
  for (iter in 1:max_tries) {
    # 1. 生成隨機正定相關矩陣 C（平均相關通常接近 0）
    A <- matrix(rnorm(df * t), nrow = df, ncol = t)
    S <- crossprod(A)
    C <- cov2cor(S)
    cbar <- mean(C[lower.tri(C)])
    
    # 若 C 的平均相關已經 >= 目標值，凸組合權重 w 會變成負值，跳過重抽
    if (cbar >= rho_target) next
    
    # 2. 求凸組合權重，使 R 的平均相關恰為 rho_target
    w <- (rho_target - cbar) / (1 - cbar)
    J <- matrix(1, nrow = t, ncol = t)
    R <- (1 - w) * C + w * J
    diag(R) <- 1
    
    off <- R[lower.tri(R)]
    if (positive_only && min(off) <= 0) next
    
    # 3. 數值上再次確認正定性（理論上凸組合必為 PSD，這裡只是保險檢查）
    eig_min <- min(eigen(R, symmetric = TRUE, only.values = TRUE)$values)
    if (eig_min > 1e-8) {
      return(R)
    }
  }
  
  stop("make_un_corr(): 嘗試 ", max_tries, " 次仍無法生成符合條件的正定相關矩陣，",
       "請考慮調高 df 或放寬 rho_target / positive_only 設定。")
}

fit_gee <- function(data, family, corstr) {
  # 動態抓取所有以 x 開頭的預測變數欄位，不再寫死 x1~x10，
  # 若資料生成端改變變數個數（例如新增/刪減 x 變數），這裡會自動跟著調整。
  x_cols      <- grep("^x", names(data), value = TRUE)
  formula_gee <- as.formula(paste("Y ~", paste(x_cols, collapse = " + ")))
  model <- geeglm(formula_gee,
                  data = data, 
                  id = ID, 
                  family = family, 
                  corstr = corstr)
  return(model)
}

fit_gpboost <- function(data, response_type = "reg",nrounds = 200) {
  x_cols <- grep("^x", names(data), value = TRUE)  
  X <- as.matrix(data[, x_cols])
  y <- data$Y
  
  gp_model <- GPModel(
    group_data = data$ID,
    likelihood = "gaussian"
  )
  
  bst <- gpboost(
    data             = X,
    label            = y,
    gp_model         = gp_model,
    nrounds          = 200,
    learning_rate    = 0.05,
    max_depth        = 5,
    min_data_in_leaf = 10,
    verbose          = 0,
    num_threads      = 1 
  )
  return(bst)
}

# 專門負責預測的函數
predict_gpboost <- function(model, df_test) {
  x_cols <- grep("^x", names(df_test), value = TRUE)  
  X_test <- as.matrix(df_test[, x_cols])
  
  pred <- predict(
    model,
    data            = X_test,
    group_data_pred = df_test$ID,
    num_threads     = 1 ,
    predict_var     = FALSE,
    pred_latent     = FALSE
  )
  return(pred$response_mean)
}

REEMforest_fixed <- function(X, Y, id, Z, iter = 100, mtry, ntree = 500, time, sto = "none", delta = 0.001) {
  if (sto != "none") {
    stop("REEMforest_fixed 目前只支援 sto = 'none'（SREEMF 使用的模式）")
  }
  
  q     <- dim(Z)[2]
  nind  <- length(unique(id))
  btilde <- matrix(0, nind, q)
  sigmahat <- 1
  Btilde <- diag(rep(1, q))
  epsilonhat <- rep(0, length(Y))
  Vrai <- NULL
  inc  <- 1
  OOB  <- NULL
  forest <- NULL
  
  for (i in 1:iter) {
    ystar <- rep(0, length(Y))
    for (k in 1:nind) {
      indiv <- which(id == unique(id)[k])
      ystar[indiv] <- Y[indiv] - Z[indiv, , drop = FALSE] %*% btilde[k, ]
    }
    
    forest <- randomForest(X, ystar, mtry = mtry, ntree = ntree,
                           importance = TRUE, keep.inbag = TRUE)
    f1     <- predict(forest, X, nodes = TRUE)
    trees  <- attributes(f1)
    OOB[i] <- forest$mse[ntree]
    inbag  <- forest$inbag
    matrice.pred <- matrix(NA, length(Y), ntree)
    
    for (k in 1:ntree) {
      Phi   <- matrix(0, length(Y), length(unique(trees$nodes[, k])))
      indii <- which(forest$forest$nodestatus[, k] == -1)
      for (l in 1:dim(Phi)[2]) {
        w <- which(trees$nodes[, k] == indii[l])
        Phi[w, l] <- 1
      }
      oobags <- unique(which(inbag[, k] == 0))
      beta <- LongituRF:::Moy(id[-oobags], Btilde, sigmahat,
                              Phi[-oobags, , drop = FALSE], ystar[-oobags],
                              Z[-oobags, , drop = FALSE])
      forest$forest$nodepred[indii, k] <- beta
      matrice.pred[oobags, k] <- Phi[oobags, ] %*% beta
    }
    
    # ── 修補的地方 ──────────────────────────────────────────────────
    fhat <- rep(NA, length(Y))
    for (k in 1:length(Y)) {
      w <- which(is.na(matrice.pred[k, ]))
      if (length(w) == 0) {
        fhat[k] <- mean(matrice.pred[k, ])              # 修正：w 為空應保留全部欄位
      } else if (length(w) == ntree) {
        fhat[k] <- predict(forest, X[k, , drop = FALSE]) # 極端情況：整棵森林都沒有這筆的 OOB 預測
      } else {
        fhat[k] <- mean(matrice.pred[k, -w])
      }
    }
    # ────────────────────────────────────────────────────────────────
    
    for (k in 1:nind) {
      indiv <- which(id == unique(id)[k])
      V <- Z[indiv, , drop = FALSE] %*% Btilde %*% t(Z[indiv, , drop = FALSE]) +
        diag(as.numeric(sigmahat), length(indiv), length(indiv))
      btilde[k, ] <- Btilde %*% t(Z[indiv, , drop = FALSE]) %*% solve(V) %*% (Y[indiv] - fhat[indiv])
      epsilonhat[indiv] <- Y[indiv] - fhat[indiv] - Z[indiv, , drop = FALSE] %*% btilde[k, ]
    }
    
    sigm <- sigmahat
    sigmahat <- LongituRF:::sig(sigma = sigmahat, id = id, Z = Z, epsilon = epsilonhat, Btilde = Btilde)
    Btilde   <- LongituRF:::bay(bhat = btilde, Bhat = Btilde, Z = Z, id = id, sigmahat = sigm)
    Vrai <- c(Vrai, LongituRF:::logV(Y, fhat, Z, time, id, Btilde, 0, sigmahat, sto))
    if (i > 1) inc <- abs((Vrai[i - 1] - Vrai[i]) / Vrai[i - 1])
    
    if (inc < delta) {
      message(paste0("stopped after ", i, " iterations."))
      sortie <- list(forest = forest, random_effects = btilde, var_random_effects = Btilde,
                     sigma = sigmahat, id_btilde = unique(id), sto = sto,
                     Vraisemblance = Vrai, id = id, time = time, OOB = OOB)
      class(sortie) <- "longituRF"
      return(sortie)
    }
  }
  sortie <- list(forest = forest, random_effects = btilde, var_random_effects = Btilde,
                 sigma = sigmahat, id_btilde = unique(id), sto = sto,
                 Vraisemblance = Vrai, id = id, time = time, OOB = OOB)
  class(sortie) <- "longituRF"
  return(sortie)
}

fit_reemf <- function(data, ntree = 50,iter = 50) {
  X_cols <- grep("^x", names(data), value = TRUE)
  X_mat  <- as.matrix(data[, X_cols])
  p      <- ncol(X_mat)
  Z_mat  <- matrix(1, nrow = nrow(data), ncol = 1)
  Y_vec  <- as.numeric(data$Y)
  
  model <- REEMforest(
    X     = X_mat,
    Y     = Y_vec,
    Z     = Z_mat,
    id    = data$ID,
    time  = data$Time,
    mtry  = floor(p / 3),
    ntree = ntree,
    iter  = iter,      
    delta = 0.01,    
    sto   = "none"
  )
  return(model)
}


#MERF
fit_merf <- function(data, ntree = 50) {
  X_cols <- grep("^x", names(data), value = TRUE)
  X_mat <- as.matrix(data[, X_cols])
  p <- ncol(X_mat)
  Z_mat <- matrix(1, nrow = nrow(data), ncol = 1)
  
  model <- MERF(
    X = X_mat,
    Y = as.numeric(data$Y),
    Z = Z_mat,
    id = data$ID,
    time = data$Time,
    mtry = ceiling(p/3),
    ntree = ntree,
    sto = "none"  
  )
  return(model)
}



cround <- function(x, digits=0) {
  posneg = sign(x)
  z = abs(x)*10^digits
  z = z + 0.5 + sqrt(.Machine$double.eps)
  z = trunc(z)
  z = z/10^digits
  z*posneg
}

#GEEB
geebm <- function(formula, id, iteration=100, feature_rate=0.5, lrate=0.1, standardize=F, data, corstr="unstructured", family = "gaussian"){
  Features <- as.vector(all.vars(formula))[-1]  
  outcome  <- as.character(formula[[2]])     
  data <- subset(data,select=(c(Features,id,outcome))) 
  mm_d    <- model.matrix(~., data = data[Features])
  term_id <- attr(mm_d, "assign")   # model.matrix 每個欄位對應 Features 的第幾個變數（intercept = 0）
  data_d  <- as.data.frame(mm_d)
  Features_d <- names(data_d)
  if(standardize==T){
    data_d[, !(names(data_d) %in% c(id,"(Intercept)"))]<-scale(data_d[, !(names(data_d) %in% c(id,"(Intercept)"))])
  }
  data_d <- cbind(data_d,data[id],data[outcome])               
  Coefficient <- setNames(data.frame(matrix(0,ncol = length(Features_d), nrow = iteration)),Features_d)  
  PredValue <- mean(data_d[[outcome]]) 
  data_d <- cbind(data_d, PredValue)
  for (m in (1:iteration)){
    R <- as.numeric(sample(1:length(Features),size=(cround(feature_rate*length(Features))),replace=F))
    # 以 assign 精確對應被抽中的變數（不再用 grep 部分比對，避免 x1 誤中 x10、age 誤中 age_sq）
    R_name_d  <- Features_d[term_id %in% R]
    data_rf_d <- data_d[, R_name_d, drop = FALSE]
    data_rf_d["Residual"] <- data_d[[outcome]]-data_d[["PredValue"]]
    data_rf_d <- cbind(data_rf_d,data_d[id[1]])
    modelGEEBM <- eval(parse(text = paste0(
      "geeglm(Residual ~ . - ", id[1], 
      ", id =", id[1], 
      ", data = data_rf_d, corstr = '", corstr, "'",
      ", family = 'gaussian')" # 在這裡加入 family 參數
    )))
    for (i in seq_along(Coefficient)) {
      if (colnames(Coefficient)[i] %in% names(coef(modelGEEBM))) {
        Coefficient[m, i] <- coef(modelGEEBM)[colnames(Coefficient)[i]] 
      }
    }
    PredResidual <- predict(modelGEEBM, newdata = data_d)
    data_d["PredResidual"] <- PredResidual
    data_d['PredValue'] <- data_d$PredValue+lrate*data_d$PredResidual
  }
  model <- structure(list(lrate=lrate, PredValue=PredValue, Features=Features, Features_d=Features_d, Coefficient=Coefficient, standardize = standardize), class = "geebm" )
  return(model)
}


predict.geebm <- function(model, data){
  data_d <- model.matrix(~., data = data[model$Features])
  if(model$standardize==T){
    data_d[,!(apply(data_d, 2, function(col) all(col == col[1])))]<-scale(data_d[,!(apply(data_d, 2, function(col) all(col == col[1])))])
  }
  prediction  <- model$PredValue+model$lrate*rowSums(data_d %*% t(model$Coefficient))
  return(as.vector(prediction))
}


geebm.importance <- function(model){
  importance <- setNames(data.frame(matrix(0,ncol = 2, nrow = length(model$Features_d))),c("Feature","Importance"))
  importance[,2] <- apply(model$Coefficient, 2, function(col) mean(col[col != 0]))
  importance[,1] <- model$Features_d
  return(importance)
}


geebm.importance.plot <- function(importance,filename="FeatureImportance.jpg"){
  plot <- ggplot(importance, aes(x = Importance, y = Feature)) +
    geom_bar(stat = "identity", fill = "#003060") +
    labs(x = NULL, y = NULL) +
    coord_flip()+
    theme(axis.line = element_line(color = "black"), 
          panel.grid.major.y = element_blank(),
          panel.background = element_blank(), 
          axis.text.y = element_text(color = "black", size = 13),
          axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, color = "black", size = 14),
          axis.ticks.x = element_line(color = "black"),
          axis.ticks.y = element_line(color = "black", size = 0.5))+ 
    scale_x_continuous(breaks = seq(-0.03, max(importance[["Importance"]]), by = 0.005))
  print(plot)
  ggsave(filename,height=3,width=5,dpi=300,units="in",scale=2, plot=plot)
}


# ── 通用內部 helper：完全自己實作，繞過原始 predict.longituRF ──────────────
.predict_longituRF_new <- function(object, X_mat, Z_mat, id, time) {
  
  # Step 1: 固定效應 f(X)，用訓練好的 forest 預測
  f <- randomForest:::predict.randomForest(object$forest, newdata = as.data.frame(X_mat))
  
  trained_ids <- object$id_btilde  
  Time_train  <- object$time        
  id_train    <- object$id          
  
  Ypred <- rep(0, length(id))
  
  for (i in seq_along(unique(id))) {
    subj <- unique(id)[i]
    w    <- which(id == subj)           
    k    <- which(trained_ids == subj)  
    
    # --- random effect 部分 ---
    if (length(k) == 0) {
      # 新受試者：random effect 補 0
      re <- rep(0, length(w))
    } else {
      re <- as.numeric(Z_mat[w, , drop = FALSE] %*% object$random_effects[k, ])
    }
    
    # --- stochastic process (omega) 部分 ---
    if (object$sto == "none") {
      omega_pred <- rep(0, length(w))
      
    } else {
      om <- which(id_train == subj)   
      
      if (length(om) == 0) {
        # 新受試者：omega 補 0
        omega_pred <- rep(0, length(w))
        
      } else {
        omega_train <- object$omega[om]
        time_train  <- Time_train[om]
        
        if (object$sto == "BM") {
          # BM：取訓練集最後時間點的 omega 值（martingale 性質）
          omega_pred <- rep(omega_train[which.max(time_train)], length(w))
          
        } else if (object$sto == "OrnUhl") {
          omega_pred <- sapply(time[w], function(t_new) {
            inf_idx <- which(time_train <= t_new)
            if (length(inf_idx) > 0) {
              t_inf <- max(time_train[inf_idx])
              omega_train[which(time_train == t_inf)] * exp(-abs(t_new - t_inf) / 2)
            } else {
              t_sup <- min(time_train)
              omega_train[which(time_train == t_sup)] * exp(-abs(t_new - t_sup) / 2)
            }
          })
          
        } else {
          # 其他 sto（BBridge、自定義函數等）：補 0 保守估計
          omega_pred <- rep(0, length(w))
        }
      }
    }
    
    Ypred[w] <- f[w] + re + omega_pred
  }
  
  return(Ypred)
}

# ── SREEMF 預測 ────────────────────────────────────────────────────────────
pred_reemf <- function(model, data) {
  X_cols <- grep("^x", names(data), value = TRUE)
  X_mat  <- as.matrix(data[, X_cols])
  Z_mat  <- matrix(1, nrow = nrow(data), ncol = 1)
  .predict_longituRF_new(model, X_mat, Z_mat, id = data$ID, time = data$Time)
}

# ── SMERF 預測 ─────────────────────────────────────────────────────────────
pred_merf <- function(model, data) {
  X_cols <- grep("^x", names(data), value = TRUE)
  X_mat  <- as.matrix(data[, X_cols])
  Z_mat  <- matrix(1, nrow = nrow(data), ncol = 1)
  .predict_longituRF_new(model, X_mat, Z_mat, id = data$ID, time = data$Time)
}

calculate_metrics <- function(actual, predicted) {
  # 確保輸入為數值向量且長度一致
  actual <- as.numeric(actual)
  predicted <- as.numeric(predicted)
  
  mse <- mean((actual - predicted)^2)
  mae <- mean(abs(actual - predicted))
  
  return(c(MSE = mse, MAE = mae))
}