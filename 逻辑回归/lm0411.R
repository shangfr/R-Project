library(pROC)
library(DMwR)
setwd('F:\\github\\R-Project\\逻辑回归')
model.df <- read.csv('customer defection data.csv',sep=',',header=T)
head(model.df)
str(model.df)
dim(model.df)
summary(model.df)
z <- model.df[,sapply(model.df, is.numeric)]
z[is.na(z)] = 0
summary(z)

exl <- names(z) %in% c('cust_id','defect')
z <- z[!exl]
head(z)
qs <- sapply(z, function(z) quantile(z,c(0.01,0.99)))
system.time(for (i in 1:ncol(z)){
  for( j in 1:nrow(z)){
    if(z[j,i] < qs[1,i]) z[j,i] = qs[1,i]
    if(z[j,i] > qs[2,i]) z[j,i] = qs[2,i]
  }
})

model_ad.df <- data.frame(cust_id=model.df$cust_id,defect=model.df$defect,z)
boxplot(model_ad.df$visit_cnt)  
set.seed(123)
s <- sample(nrow(model_ad.df),floor(nrow(model_ad.df)*0.7),replace = F)
train_df <- model_ad.df[s,]
test_df <- model_ad.df[-s,]

n <- names(train_df[-c(1,34)])
f <- as.formula(paste('defect ~',paste(n[!n %in% 'defect'],collapse = ' + ')))
model_full <- glm(f,data=train_df[-c(1,34)],family = binomial)
summary(model_full)

step <- step(model_full,direction = 'forward')
summary(step)
pred <- predict(step,test_df,type='response')
head(pred)
fitted.r <- ifelse(pred>0.5,1,0)
accuracy <- table(fitted.r,test_df$defect)
misClassificError <- mean(fitted.r != test_df$defect)
roc <- roc(test_df$defect,pred)
roc
plot(roc)
library(pROC)
library(DMwR)
cs.df <- read.csv('E:\\Udacity\\Data Analysis High\\R\\R_Study\\第二天数据\\cs-data.csv',header=T,sep=',')
summary(cs.df)
# SeriousDlqin2yrs 超过90天的逾期欠款
# RevolvingUtilizationOfUnsecuredLines 无担保贷款的循环利用,除了车,房除以信用额度的综合的无分期债务的信用卡贷款
# age 贷款人年龄
# NumberOfTime30-59DaysPastDueNotWorse 30~59天逾期次数
# DebtRatio 负债比例
# MonthlyIncome 月收入
# NumberOfOpenCreditLinesAndLoans 开放式和信贷的数量
# NumberOfTimes90DaysLate 大于等于90天逾期的次数
# NumberRealEstateLoansOrLines 不动产的数量
# NumberOfTime60-89DaysPastDueNotWorse 60~90天逾期次数
# NumberOfDependents 不包括本人的家属数量
# 使用knn邻近算法,补充缺失的月收入
cs.df_imp <- knnImputation(cs.df,k=3,meth = 'weighAvg')
#去除掉 30~60天逾期超过80的极大值
cs.df_imp <- cs.df_imp[-which(cs.df_imp$NumberOfTime30.59DaysPastDueNotWorse>80)]
# 去除掉负债比大于10000的极值
cs.df_imp <- cs.df_imp[-which(cs.df_imp$DebtRatio > 100000)]
# 去除掉月收入大于50万的极值
cs.df_imp <- cs.df_imp[-which(cs.df_imp$MonthlyIncome > 500000)]
set.seed(123)
# 将数据集分成训练集和测试集,防止过拟合
s <- sample(nrow(cs.df_imp),floor(nrow(cs.df_imp)*0.7),replace = F)
cs.train <- cs.df_imp[s,]
cs.test <- cs.df_imp[-s,]
# 使用逻辑线性回归生成全量模型
# family=binomia表示使用二项分布
# maxit=1000 表示需要拟合1000次
model_full <- glm(SeriousDlqin2yrs~.,data=cs.train,family=binomial,maxit=1000)
# 使用回归的方式找出最小的AIC的值
step <- step(model_full,direction='both')
summary(step)

pred <- predict(step,cs.test,type = 'response')
fitted.r <- ifelse(pred>0.5,1,0)
accuracy <- table(fitted.r,cs.test$SeriousDlqin2yrs)
misClasificError <- mean(fitted.r!=cs.test$SeriousDlqin2yrs)
roc <- roc(cs.test$SeriousDlqin2yrs,pred)
plot(roc)
roc
# 修改模型
table(cs.train$SeriousDlqin2yrs)
prop.table(table(cs.train$SeriousDlqin2yrs))
cs.train$SeriousDlqin2yrs <- as.factor(cs.train$SeriousDlqin2yrs)
# 采用bootstrasp自助抽样法,目的:减小0的个数,增加1的个数,再平衡模型
trainSplit <- SMOTE(SeriousDlqin2yrs~.,cs.train,perc.over = 30,perc.under = 550)
cs.train$SeriousDlqin2yrs <- as.numeric(cs.train$SeriousDlqin2yrs)
prop.table(table(trainSplit$SeriousDlqin2yrs))
model_full =  glm(SeriousDlqin2yrs~.,data=trainSplit,family=binomial,maxit=1000)

step = step(model_full,direction = "both")
summary(step)
pred = predict(step,cs.test,type="response")


fitted.r=ifelse(pred>0.5,1,0)
accuracy = table(fitted.r,cs.test$SeriousDlqin2yrs)

misClasificError = mean(fitted.r!=cs.test$SeriousDlqin2yrs)

roc = roc(cs.test$SeriousDlqin2yrs,pred)
plot(roc)
roc