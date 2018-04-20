# 神经网络


#3个神经元,学习速率0.01
nnetOneLayer <- function(x,y,numberofUnits = 20,alpha = 0.01,iter=10000){
  x <- as.matrix(x)
  # x的特征数量
  col <- ncol(x)
  row <- nrow(x)
  #输入层的权重初始化 nrow 表示特征数
  syn0 <- matrix(rnorm(numberofUnits * col ),ncol=col,nrow=numberofUnits)
  #隐层的权重初始化nrow=1,即隐层只有一个神经元
  syn1 <- matrix(rnorm(numberofUnits),ncol=numberofUnits,nrow=1)
  cost <- matrix(rep(NA,iter*row),row,iter)
  for (i in 1:iter){
    #输入层到隐层的值 激励函数sigmoid=1/(1+exp(-y)) ,其导数为sigmoid*(1-sigmoid)
    #加入非线性激励函数后，线性分类器变成非线性。
    l1 <- 1/(1+exp(-(x %*% t(syn0) )))
    #隐层到输出层的值
    l2 <- 1/(1+exp(-l1  %*% t(syn1) ))
    #输出层误差 损失函数
    errorOutput <- l2 - y
    #隐层误差 权重矩阵的偏导数=ds*w*s*(1-s) 理论推导见：Sigmoid函数与损失函数求导
    #https://blog.csdn.net/zhishengqianjun/article/details/75303820
    errorHidder <- errorOutput %*% syn1 * l1  * (1-l1 )
    
    #梯度下降（Gradient Descent）
    syn1 <- syn1 -  t(alpha * t(l1) %*% errorOutput )
    syn0 <- syn0 - t(alpha * t(x) %*% errorHidder) 
    
    
    cost[,i] <- errorOutput
  }
  
  plot(cost[1,],type = 'l',ylim = c(-1,1))
  lines(cost[2,],col = 'blue',lwd = 2)
  lines(cost[3,],col = 'red',lwd = 2)
  lines(cost[4,],col = 'blue',lwd = 2)
  cbind(y,l2)
# heatmap(syn0,main = "gray2",scale="column",revC=T, Rowv=NA, Colv=NA,labRow=NA, labCol=NA)

}

x <- t(data.frame(c(0,0,1),c(0,1,1),c(1,0,1),c(1,1,1)))
y <- c(0,1,1,0)
nnetOneLayer(x,y)

