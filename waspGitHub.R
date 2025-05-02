#WASP methodology

library(dplyr)
library(tibble)
library(ggplot2)
library(qape)
library(reshape2)
library(ks)
library(e1071)
library(mgcv)
library(rpart)
library(scales)


# Original dataset ----
data(invData)
head(invData)

dataS <- invData |> select(-c(NUTS4, NUTS4type)) |> as_tibble()
dataS$year <- as.numeric(as.character(dataS$year))
names(dataS)
dim(dataS)

# EDA - Distribution of response variable investments ----
#resYears <- 
dataS %>% group_by(year) %>% 
  summarise(n=n(),  
    median = median(investments), total = sum(investments))

dataS %>% group_by(NUTS2) %>% 
  summarise(n=n(), total = sum(investments), 
            median = median(investments))

# distribution plots ----
cut <- quantile(dataS$investments, p=c(0.9, 0.95, 0.99, 0.995))
dataScut <- subset(dataS, dataS$investments <= cut[2])

dataScut %>%  ggplot(aes(x = investments)) +
  geom_histogram(aes(y = ..density..), fill = 'grey', alpha = 0.99) +
  geom_density(fill = 'lightblue', alpha = 0.3) + theme_bw() +
  facet_wrap(~year) +
  ylab('') + xlab('investments') +
  geom_vline(xintercept = median(dataS$investments), linetype = 2) +
  annotate("text", x=median(dataS$investments)*1.3, y=0.005,
           label= "Median", size = 4) +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))
 

ggplot(dataScut, aes(x = NUTS2, y = investments)) +
    geom_boxplot(fill = 'lightblue', alpha = 0.3) +
    theme_bw() + ylab('') + xlab('investments in NUTS2') + facet_wrap(~year)

# Input WASP ----
regS <- dataS |> select(-investments)
regR <- subset(dataS, year == 2018) |> select(-investments)
regR$year <- 2019
reg <- bind_rows(regS, regR)

regF <- 'NUTS2+log(newly_registered)'
modelF <- formula('investments ~ NUTS2+log(newly_registered)')
modelFlog <- formula('log(investments) ~ NUTS2+log(newly_registered)')

regFspline <- 'NUTS2+s(log(newly_registered))'
modelFSpline <- formula('investments ~ NUTS2+s(log(newly_registered))')
modelFSplinelog <- formula('log(investments) ~ NUTS2+s(log(newly_registered))')
con <- c(rep(1, nrow(dataS)), rep(0, nrow(regR))) 

# Theta function
thetaFun <- function(x) {
  return(data.frame(median = median(x), sum = sum(x)))
  }
thetaFun(dataS$investments)

# Accuracy fun
rmseFun <- function(y,yp){mean((as.numeric(y - yp))^2) |> sqrt() |> round(2)}
qapeFun <- function(y,yp, prob){quantile(abs(as.numeric(y - yp)), prob, names = F) |> round(2)}

# Prediction models - sample data
modelLogN <- lm(modelFlog, data = dataS)
modelGLMlog <- glm(modelFlog, family = Gamma(link = "log"), data = dataS)
modelGAMlog <- gam(modelFSplinelog, family = inverse.gaussian(link = 'log'), data = dataS)
modelTreelog <- rpart(modelFlog, data = dataS)
modelSVMlog <- svm(modelFlog, type = 'eps-regression',  kernel = 'linear', data = dataS)
modelSVMpolylog <- svm(modelFlog, type = 'eps-regression',  kernel = 'polynomial', data = dataS)

# Fitted values sample data
muSLogN <- predict(modelLogN, dataS, type = 'response')
muSGLMlog <- predict(modelGLMlog, dataS, type = 'response')
muSGAMlog <- predict(modelGAMlog, dataS, type = 'response')
muSTreelog <- predict(modelTreelog, dataS)
muSSVMlog <- predict(modelSVMlog, dataS)
muSSVMpolylog <- predict(modelSVMpolylog, dataS)


# Kernel bootstrap functions
KERNELLogN <- function(){
  reszty <- log(dataS$investments) - muSLogN
  fhat <- kde(x = reszty)
  residSampl <- rkde(nrow(reg), fhat, positive=FALSE)
  residSampl <- residSampl - mean(residSampl)
  Ysim <- (predict(modelLogN, reg, type = 'response') + residSampl)
  YsimS <- Ysim[con == 1]
  model <- lm(formula(paste('YsimS~', regF)), data = c(YsimS, regS))
  YpredR <- predict(modelLogN, regR, type = 'response')
  return(list(YpredRlog = YpredR, Ysimlog = Ysim))
  }

KERNELGLM <- function(){
  reszty <- log(dataS$investments) - muSGLMlog 
  fhat <- kde(x = reszty)
  residSampl <- rkde(nrow(reg), fhat, positive=FALSE)
  residSampl <- residSampl - mean(residSampl)
  Ysim <- (predict(modelGLMlog, reg, type = 'response') + residSampl)
  YsimS <- Ysim[con == 1]
  model <- glm(formula(paste('YsimS~', regF)), family = Gamma(link = "log"), 
               data = c(YsimS, regS))
  YpredR <- predict(model, regR, type = 'response')
  return(list(YpredRlog = YpredR, Ysimlog = Ysim))
}

KERNELGAM <- function(){
  reszty <- log(dataS$investments) - muSGAMlog
  fhat <- kde(x = as.numeric(reszty))
  residSampl <- rkde(nrow(reg), fhat, positive=FALSE)
  residSampl <- residSampl - mean(residSampl)
  Ysim <- predict(modelGAMlog, reg, type = 'response') + residSampl
  YsimS <- Ysim[con == 1]
  model <- gam(formula(paste('YsimS~', regFspline)), family  = inverse.gaussian(link = 'log'), data=c(YsimS, regS))
  YpredR <-  predict(model, regR, type = 'response')
  return(list(YpredRlog = YpredR, Ysimlog = Ysim))
}

KERNELTree <- function(){
  reszty <- log(dataS$investments) - muSTreelog
  fhat <- kde(x = reszty)
  residSampl <- rkde(nrow(reg), fhat, positive=FALSE) 
  residSampl <- residSampl - mean(residSampl)
  Ysim <- predict(modelTreelog, reg) + residSampl
  YsimS <- Ysim[con == 1]
  model <- rpart(formula(paste('YsimS~', regF)), data=c(YsimS, regS))
  YpredR <-  predict(model, regR)
  return(list(YpredRlog = YpredR, Ysimlog = Ysim))
}

KERNELSVM <- function(){
  reszty <- log(dataS$investments) - muSSVMlog
  fhat <- kde(x = reszty)
  residSampl <- rkde(nrow(reg), fhat, positive=FALSE) 
  residSampl <- residSampl - mean(residSampl)
  Ysim <- predict(modelSVMlog, reg) + residSampl
  YsimS <- Ysim[con == 1]
  model <- svm(formula(paste('YsimS~', regF)), type = 'eps-regression',  kernel = 'linear', data=c(YsimS, regS))
  YpredR <-  predict(model, regR)
  return(list(YpredRlog = YpredR, Ysimlog = Ysim))
}

KERNELSVMpoly <- function(){
  reszty <- log(dataS$investments) - muSSVMpolylog
  fhat <- kde(x = reszty)
  residSampl <- rkde(nrow(reg), fhat, positive=F)
  residSampl <- residSampl - mean(residSampl)
  Ysim <- predict(modelSVMpolylog, reg) + residSampl
  YsimS <- Ysim[con == 1]
  model <- svm(formula(paste('YsimS~', regF)), type = 'eps-regression',  kernel = 'polynomial', data=c(YsimS, regS))
  YpredR <-  predict(model, regR)
  return(list(YpredRlog = YpredR, Ysimlog = Ysim))
}

# WASP MONTE CARLO ----
set.seed(247)
#nsim <- 5000
nsim <- 2

llgenerator <- lapply(1:nsim, function(i){
  print(i)
  return(list(
    genLogN = KERNELLogN(),
    genGLM = KERNELGLM(),
    genGAM = KERNELGAM(),
    genTree = KERNELTree(),
    genSVM = KERNELSVM(),
    genSVMpoly = KERNELSVMpoly()
  )) 
})

lltheta <- lapply(1:nsim, function(i){
  print(i)
  genLogN <- llgenerator[[i]]$genLogN
  genGLM <- llgenerator[[i]]$genGLM
  genGAM <- llgenerator[[i]]$genGAM
  genTree <- llgenerator[[i]]$genTree
  genSVM <- llgenerator[[i]]$genSVM
  genSVMpoly <- llgenerator[[i]]$genSVMpoly
  
  #Ysim nr Modelgen
  Ysim1log <- genLogN$Ysimlog
  Ysim2log <- genGLM$Ysimlog
  Ysim3log <- genGAM$Ysimlog
  Ysim4log <- genTree$Ysimlog
  Ysim5log <- genSVM$Ysimlog
  Ysim6log <- genSVMpoly$Ysimlog
  
  
  YsimS1log <- Ysim1log[con == 1]
  YsimS2log <- Ysim2log[con == 1] |> as.numeric()
  YsimS3log <- Ysim3log[con == 1] |> as.numeric()
  YsimS4log <- Ysim4log[con == 1]
  YsimS5log <- Ysim5log[con == 1]
  YsimS6log <- Ysim6log[con == 1]
  
  #thetaSim nr Modelgen
  thetaSim1 <- thetaFun(exp(Ysim1log))
  thetaSim2 <- thetaFun(exp(Ysim2log))
  thetaSim3 <- thetaFun(exp(Ysim3log))
  thetaSim4 <- thetaFun(exp(Ysim4log))
  thetaSim5 <- thetaFun(exp(Ysim5log))
  thetaSim6 <- thetaFun(exp(Ysim6log))
  
  #thetaPred  nr (Modelpred, Modelgen) ----
  thetaPred11 <- c(YsimS1log, genLogN$YpredR) |> exp() |> thetaFun()
  thetaPred12 <- lm(formula(paste('YsimS2log ~',regF)), c(YsimS2log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred13 <- lm(formula(paste('YsimS3log ~',regF)), c(YsimS3log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred14 <- lm(formula(paste('YsimS4log ~',regF)), c(YsimS4log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred15 <- lm(formula(paste('YsimS5log ~',regF)), c(YsimS5log, regS)) |> 
    predict(regR) |> exp() |>  thetaFun()
  thetaPred16 <- lm(formula(paste('YsimS6log ~',regF)), c(YsimS6log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  
  
  thetaPred21 <- gam(formula(paste('YsimS2log ~',regFspline)), family  = inverse.gaussian(link = 'log'),
                     data = c(YsimS1log, regS)) |> predict(regR, type = 'response') |> exp() |> thetaFun()
  thetaPred22 <- c(YsimS2log, genGLM$YpredR) |> exp() |> thetaFun()
  thetaPred23 <- gam(formula(paste('YsimS3log ~',regFspline)), family  = inverse.gaussian(link = 'log'), 
                     data = c(YsimS3log, regS)) |> predict(regR, type = 'response') |> exp()|> thetaFun()
  thetaPred24 <- gam(formula(paste('YsimS4log ~',regFspline)), family  = inverse.gaussian(link = 'log'),
                     data = c(YsimS4log, regS)) |> predict(regR, type = 'response') |> exp()|> thetaFun()
  thetaPred25 <- gam(formula(paste('YsimS3log ~',regFspline)), family  = inverse.gaussian(link = 'log'),
                     data = c(YsimS5log, regS)) |> predict(regR, type = 'response') |> exp()|> thetaFun()
  thetaPred26 <- gam(formula(paste('YsimS3log ~',regFspline)), family  = inverse.gaussian(link = 'log'),
                     data = c(YsimS6log, regS)) |> predict(regR, type = 'response') |> exp()|> thetaFun()
  
  thetaPred31 <- glm(formula(paste('YsimS1log ~',regF)), family = Gamma(link = "log"), c(YsimS1log, regS)) |> 
    predict(regR, type = 'response') |> exp() |> thetaFun()
  thetaPred32 <- glm(formula(paste('YsimS2log ~',regF)), family = Gamma(link = "log"), c(YsimS2log, regS)) |> 
    predict(regR, type = 'response') |> exp() |>thetaFun() 
  thetaPred33 <- c(YsimS3log, genGAM$YpredR) |> exp() |> thetaFun()
  thetaPred34 <- glm(formula(paste('YsimS4log ~',regF)), family = Gamma(link = "log"), c(YsimS4log, regS)) |> 
    predict(regR, type = 'response') |> exp() |>thetaFun()
  thetaPred35 <- glm(formula(paste('YsimS5log ~',regF)), family = Gamma(link = "log"), c(YsimS5log, regS)) |> 
    predict(regR, type = 'response') |> exp() |>thetaFun()
  thetaPred36 <- glm(formula(paste('YsimS6log ~',regF)), family = Gamma(link = "log"), c(YsimS6log, regS)) |> 
    predict(regR, type = 'response') |> exp() |>thetaFun()
  
  
  thetaPred41 <- rpart(formula(paste('YsimS1log ~',regF)), c(YsimS1log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred42 <- rpart(formula(paste('YsimS2log ~',regF)), c(YsimS2log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred43 <- rpart(formula(paste('YsimS3log ~',regF)), c(YsimS3log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred44 <- c(YsimS4log, genTree$YpredRlog) |> exp() |> thetaFun()
  thetaPred45 <- rpart(formula(paste('YsimS5log ~',regF)), c(YsimS5log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred46 <- rpart(formula(paste('YsimS6log ~',regF)), c(YsimS6log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  
  
  thetaPred51 <- svm(formula(paste('YsimS1log ~',regF)), type = 'eps-regression',  kernel = 'linear', c(YsimS1log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred52 <- svm(formula(paste('YsimS2log ~',regF)), type = 'eps-regression',  kernel = 'linear', c(YsimS2log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred53 <- svm(formula(paste('YsimS3log ~',regF)), type = 'eps-regression',  kernel = 'linear', c(YsimS3log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred54 <- svm(formula(paste('YsimS4log ~',regF)), type = 'eps-regression',  kernel = 'linear', c(YsimS4log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred55 <- c(YsimS5log, genSVM$YpredR) |> thetaFun()
  thetaPred56 <- svm(formula(paste('YsimS6log ~',regF)), type = 'eps-regression',  kernel = 'linear', c(YsimS6log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  
  thetaPred61 <- svm(formula(paste('YsimS1log ~',regF)), type = 'eps-regression',  kernel = 'polynomial', c(YsimS1log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred62 <- svm(formula(paste('YsimS2log ~',regF)), type = 'eps-regression',  kernel = 'polynomial', c(YsimS2log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred63 <- svm(formula(paste('YsimS3log ~',regF)), type = 'eps-regression',  kernel = 'polynomial', c(YsimS3log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred64 <- svm(formula(paste('YsimS4log ~',regF)), type = 'eps-regression',  kernel = 'polynomial', c(YsimS4log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred65 <- svm(formula(paste('YsimS5log ~',regF)), type = 'eps-regression',  kernel = 'polynomial', c(YsimS5log, regS)) |> 
    predict(regR) |> exp() |> thetaFun()
  thetaPred66 <- c(YsimS6log, genSVMpoly$YpredRlog) |> exp() |> thetaFun()
  
  return(list(
    thetaPred1 = data.frame(thetaPred11 = thetaPred11, thetaPred12 = thetaPred12, thetaPred13 = thetaPred13, thetaPred14 = thetaPred14, thetaPred15 = thetaPred15, thetaPred16 = thetaPred16),
    thetaPred2 = data.frame(thetaPred21 = thetaPred21, thetaPred22 = thetaPred22, thetaPred23 = thetaPred23, thetaPred24 = thetaPred24, thetaPred25 = thetaPred25, thetaPred26 = thetaPred26),
    thetaPred3 = data.frame(thetaPred31 = thetaPred31, thetaPred32 = thetaPred32, thetaPred33 = thetaPred33, thetaPred34 = thetaPred34, thetaPred35 = thetaPred35, thetaPred36 = thetaPred36),
    thetaPred4 = data.frame(thetaPred41 = thetaPred41, thetaPred42 = thetaPred42, thetaPred43 = thetaPred43, thetaPred44 = thetaPred44, thetaPred45 = thetaPred45, thetaPred46 = thetaPred46),
    thetaPred5 = data.frame(thetaPred51 = thetaPred51, thetaPred52 = thetaPred52, thetaPred53 = thetaPred53, thetaPred54 = thetaPred54, thetaPred55 = thetaPred55, thetaPred56 = thetaPred56),
    thetaPred6 = data.frame(thetaPred61 = thetaPred61, thetaPred62 = thetaPred62, thetaPred63 = thetaPred63, thetaPred64 = thetaPred64, thetaPred65 = thetaPred65, thetaPred66 = thetaPred66),
    
    thetaSim = data.frame(thetaSim1 = thetaSim1, thetaSim2=thetaSim2, thetaSim3=thetaSim3, 
                          thetaSim4=thetaSim4, thetaSim5=thetaSim5, thetaSim6=thetaSim6)
    
    ))
})

# MC simulation results ---
thetaSim <- do.call(rbind, lapply(lltheta, function(x) x$thetaSim))
namess <- names(thetaSim) 

thetaPred1 <- do.call(rbind, lapply(lltheta, function(x) x$thetaPred1))
thetaPred2 <- do.call(rbind, lapply(lltheta, function(x) x$thetaPred2))
thetaPred3 <- do.call(rbind, lapply(lltheta, function(x) x$thetaPred3))
thetaPred4 <- do.call(rbind, lapply(lltheta, function(x) x$thetaPred4))
thetaPred5 <- do.call(rbind, lapply(lltheta, function(x) x$thetaPred5))
thetaPred6 <- do.call(rbind, lapply(lltheta, function(x) x$thetaPred6))

rmse1 = sapply(1:ncol(thetaSim), function(i) rmseFun(thetaSim[ ,i], thetaPred1[ ,i]))
qape1 = sapply(1:ncol(thetaSim), function(i) qapeFun(thetaSim[ ,i], thetaPred1[ ,i], c(0.5, 0.99)))
accuRow1 = rbind(rmse1, qape1) |> as.data.frame()
names(accuRow1) <- namess

rmse2 = sapply(1:ncol(thetaSim), function(i) rmseFun(thetaSim[ ,i], thetaPred2[ ,i]))
qape2 = sapply(1:ncol(thetaSim), function(i) qapeFun(thetaSim[ ,i], thetaPred2[ ,i], c(0.5, 0.99)))
accuRow2 = rbind(rmse2, qape2) |> as.data.frame()

rmse3 = sapply(1:ncol(thetaSim), function(i) rmseFun(thetaSim[ ,i], thetaPred3[ ,i]))
qape3 = sapply(1:ncol(thetaSim), function(i) qapeFun(thetaSim[ ,i], thetaPred3[ ,i], c(0.5, 0.99)))
accuRow3 = rbind(rmse3, qape3) |> as.data.frame()

rmse4 = sapply(1:ncol(thetaSim), function(i) rmseFun(thetaSim[ ,i], thetaPred4[ ,i]))
qape4 = sapply(1:ncol(thetaSim), function(i) qapeFun(thetaSim[ ,i], thetaPred4[ ,i], c(0.5, 0.99)))
accuRow4 = rbind(rmse4, qape4) |> as.data.frame()

rmse5 = sapply(1:ncol(thetaSim), function(i) rmseFun(thetaSim[ ,i], thetaPred5[ ,i]))
qape5 = sapply(1:ncol(thetaSim), function(i) qapeFun(thetaSim[ ,i], thetaPred5[ ,i], c(0.5, 0.99)))
accuRow5 = rbind(rmse5, qape5) |> as.data.frame()

rmse6 = sapply(1:ncol(thetaSim), function(i) rmseFun(thetaSim[ ,i], thetaPred6[ ,i]))
qape6 = sapply(1:ncol(thetaSim), function(i) qapeFun(thetaSim[ ,i], thetaPred6[ ,i], c(0.5, 0.99)))
accuRow6 = rbind(rmse6, qape6) |> as.data.frame()


# Accuracy matrix  ----
accuAll <- cbind(melt(accuRow1), melt(accuRow2)$value, melt(accuRow3)$value, 
                 melt(accuRow4)$value, melt(accuRow5)$value, melt(accuRow6)$value) |> as_tibble()
names(accuAll) <- c('accu', 'strategy1', 'strategy2', 'strategy3', 'strategy4', 'strategy5', 'strategy6')
accuAll

# Voting ----
Wmatrix <- t(apply(accuAll[-1], 1, function(x) ifelse(x == min(x), 1, 0))) |> as_tibble()
vote <-   Wmatrix %>% apply(.,2,sum) %>% as.numeric() %>% data.frame()
vote$Prediction_strategy <- colnames(accuAll[-1])
names(vote)[1] <- 'sum'
vote

# Results
vote[which.max(vote$sum), ]

# Prediction GLM-Gamma strategy3 ----
pred3 <- predict(modelGLM, regR, type = 'response') |> thetaFun()
resYears[7, ] <- c(2019, 380, pred3[1], pred3[2])

