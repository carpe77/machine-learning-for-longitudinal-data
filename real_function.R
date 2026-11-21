library(MASS)
library(gpboost)
library(LongituRF)
library(geepack)
library(ggplot2)
# Predictor columns: everything except ID, Y and Time (Time is only used as the `time` argument of MERF/REEMforest)
get_x_cols <- function(d) setdiff(names(d), c("ID", "Y", "Time"))
logistic <- function(x){ (1+exp(-x))^(-1) }

fit_gee <- function(data,
                    family = gaussian,
                    corstr = "independence") {
  
  X_cols <- get_x_cols(data)
  
  formula_gee <- as.formula(
    paste("Y ~", paste(X_cols, collapse = " + "))
  )
  
  model <- geeglm(
    formula_gee,
    data = data,
    id = ID,
    family = family,
    corstr = corstr
  )
  
  return(model)
}

fit_gpboost <- function(data, response_type = "reg") {
  
  X_cols <- get_x_cols(data)
  
  X <- as.matrix(data[, X_cols])
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
  
  X_cols <- get_x_cols(df_test)
  
  X_test <- as.matrix(df_test[, X_cols])
  
  pred <- predict(
    model,
    data            = X_test,
    group_data_pred = df_test$ID,
    predict_var     = FALSE,
    pred_latent     = FALSE
  )
  
  return(pred$response_mean)
}

#SREEMF
fit_sreemf <- function(data, ntree = 100, iter = 50, delta = 0.01) {
  
  X_cols <- get_x_cols(data)
  
  X_mat <- as.matrix(data[, X_cols])
  
  p <- ncol(X_mat)
  
  Y_vec <- as.numeric(data$Y)
  
  Z_mat <- matrix(1, nrow = nrow(data), ncol = 1)
  
  model <- REEMforest(
    X = X_mat,
    Y = Y_vec,
    Z = Z_mat,
    id = data$ID,
    time = data$Time,
    mtry = floor(p / 3),
    ntree = ntree,
    iter = iter,
    delta = delta,
    sto = "none"
  )
  
  return(model)
}

#SMERF
fit_smerf <- function(data, ntree = 100) {
  
  X_cols <- get_x_cols(data)
  
  X_mat <- as.matrix(data[, X_cols])
  
  p <- ncol(X_mat)
  
  Z_mat <- matrix(1, nrow = nrow(data), ncol = 1)
  
  model <- MERF(
    X = X_mat,
    Y = as.numeric(data$Y),
    Z = Z_mat,
    id = data$ID,
    time = data$Time,
    mtry = ceiling(p / 3),
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
geebm <- function(formula, id, iteration=100, feature_rate=0.5, lrate=0.1, standardize=F, data, corstr="independence", family){
  Features <- as.vector(all.vars(formula))[-1]  
  outcome  <- as.character(formula[[2]])     
  data <- subset(data,select=(c(Features,id,outcome))) 
  data_d <- as.data.frame(model.matrix(~., data=data[Features]))  
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
    R_name_d <- grep(paste(Features[R], collapse = "|"), colnames(data_d), value = TRUE)
    data_rf_d <- data_d[,R_name_d]
    data_rf_d["Residual"] <- data_d[[outcome]]-data_d[["PredValue"]]
    data_rf_d <- cbind(data_rf_d,data_d[id[1]])
    modelGEEBM <- eval(parse(text = paste0(
      "geeglm(Residual ~ . - ", id[1], 
      ", id =", id[1], 
      ", data = data_rf_d, corstr = '", corstr, "'",
      ", family = '", family, "')" # 在這裡加入 family 參數
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
  
  trained_ids <- object$id_btilde   # 訓練集中有 random_effects 的 ID 清單
  Time_train  <- object$time        # 訓練集的時間向量（omega 對應用）
  id_train    <- object$id          # 訓練集完整 id 向量（omega 索引用）
  
  Ypred <- rep(0, length(id))
  
  for (i in seq_along(unique(id))) {
    subj <- unique(id)[i]
    w    <- which(id == subj)           # 測試集中該受試者的列索引
    k    <- which(trained_ids == subj)  # 在 random_effects 的位置
    
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
      om <- which(id_train == subj)   # 訓練集中該受試者的 omega 索引
      
      if (length(om) == 0) {
        # 新受試者：omega 補 0
        omega_pred <- rep(0, length(w))
        
      } else {
        # 舊受試者：用訓練集的 omega 外推到測試集時間點
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
pred_sreemf <- function(model, data) {
  
  X_cols <- get_x_cols(data)
  
  X_mat <- as.matrix(data[, X_cols])
  
  Z_mat <- matrix(1, nrow = nrow(data), ncol = 1)
  
  .predict_longituRF_new(
    model,
    X_mat,
    Z_mat,
    id = data$ID,
    time = data$Time
  )
}

# ── SMERF 預測 ─────────────────────────────────────────────────────────────
pred_smerf <- function(model, data) {
  
  X_cols <- get_x_cols(data)
  
  X_mat <- as.matrix(data[, X_cols])
  
  Z_mat <- matrix(1, nrow = nrow(data), ncol = 1)
  
  .predict_longituRF_new(
    model,
    X_mat,
    Z_mat,
    id = data$ID,
    time = data$Time
  )
}
calculate_metrics <- function(actual, predicted) {
  # 確保輸入為數值向量且長度一致
  actual <- as.numeric(actual)
  predicted <- as.numeric(predicted)
  
  mse <- mean((actual - predicted)^2)
  mae <- mean(abs(actual - predicted))
  
  return(c(MSE = mse, MAE = mae))
}