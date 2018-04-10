library(ROCR)
??ROCR
data(ROCR.simple)
pred <- prediction(ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

auc = performance(pred,"auc")

auc = unlist(slot(auc,'y.values'))