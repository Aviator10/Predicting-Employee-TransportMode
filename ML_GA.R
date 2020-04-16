carData<-read.csv("Cars.csv",header = TRUE)
dim(carData)
str(carData)
head(carData)
tail(carData)
seed=6789

#NULL value check and treatment
sum(is.na(carData))
sapply(carData, function(x) sum(is.na(x)))
sapply(carData, function(x) length(unique(x)))
carData<-na.omit(carData)#Removed 1 row for NA MBA

summary(carData)

##transforming into categorical variables
carData$MBA<-as.factor(carData$MBA)
carData$license<-as.factor(carData$license)
carData$Engineer<-as.factor(carData$Engineer)
str(carData)

#Adding a new column for prediction and removing original Transport col
carData$Uses_Car<-ifelse(carData$Transport=='Car',1,0)
carData$Uses_Car<-as.factor(carData$Uses_Car)
carDataNEW<-carData[,-9]

summary(carDataNEW)


#Check for outliers
boxplot(carDataNEW[,c(1,5,6,7)],col = "magenta", main='Check for Outliers')

##Univariate Analysis
#Histograms
carDataEDA<-carDataNEW[,c(1,5,6,7)]
carDataEDA<-as.data.frame(carDataEDA)
par(mfrow=c(2,2))
for (i in (1:4)) {
  hist (carDataEDA[,i],
        main = 'Histograms',xlab=colnames(carDataEDA[i]),ylab=NULL,col=c('blue','green')
  )
}

#ScatterPlots
for (i in (1:4)) {
  plot (carDataEDA[,i],
        main = 'Scatter Plots',ylab=colnames(carDataEDA[i]),col=c('yellow','red')
  )
}

dev.off()

##Bivariate Analysis
histogram(~carDataNEW$Uses_Car|factor(carDataNEW$MBA),data = carDataNEW,main="Car Usage wrt MBA")
histogram(~carDataNEW$Uses_Car|factor(carDataNEW$Engineer),data = carDataNEW,main="Car Usage wrt Engineer")
histogram(~carDataNEW$Uses_Car|factor(carDataNEW$Gender),data = carDataNEW,main="Car Usage wrt Gender")
histogram(~carDataNEW$Uses_Car|factor(carDataNEW$license),data = carDataNEW,main="Car Usage wrt license")

boxplot(carDataNEW$Salary~carDataNEW$Engineer, main = "Salary vs Engineer")
boxplot(carDataNEW$Salary~carDataNEW$MBA, main = "Salary vs MBA")
boxplot(carDataNEW$Salary~carDataNEW$Uses_Car, main = "Salary vs Car Usage")
boxplot(carDataNEW$Distance~carDataNEW$Uses_Car, main = "Distance vs Car Usage")
boxplot(carDataNEW$Age~carDataNEW$Uses_Car, main = "Age vs Car Usage")

sum(carDataNEW$Uses_Car == 1)/nrow(carDataNEW)
table(carDataNEW$Uses_Car)
pie(table(carDataNEW$Uses_Car))

##Checking multicollinearity..ignoring categorical variables
carDataNEW.scatter<-subset(carDataNEW[,c(1,5,6,7)])
cor.plot(carDataNEW.scatter,numbers = TRUE,xlas=2)
GGally::ggpairs(carDataNEW[,c(1,5,6,7)], mapping = aes(color = carDataNEW$Uses_Car))

attach(carDataNEW)
#Check Multicollinearity
#logistic : Model 1
LRModel_1=glm(Uses_Car~., data = carDataNEW, family = binomial(link="logit"))
#Check multicollienearity
vif(LRModel_1)
#Vif for Age and Work.Exp is high, we can drop work.exp and check
#logistic : Model 2
LRModel_2=glm(Uses_Car~Age+Gender+Engineer+MBA+Salary+Distance+license, data
              = carDataNEW, family = binomial(link="logit"))
#Check multicollienearity
vif(LRModel_2)
#Drop Work Exp
carDataFinal=carDataNEW[,-5]
summary(carDataFinal)
sum(carDataFinal$Uses_Car == 1)/nrow(carDataNEW)
table(carDataFinal$Uses_Car)
pie(table(carDataFinal$Uses_Car))


#########Checking for imbalance data and treating with SMOTE##################
#set seed
set.seed(seed)
#70:30 ratio data splitting
# Get 70% of the sample size
splitLR = sample.split(carDataFinal$Uses_Car, SplitRatio = 0.7)
trainDataLR<-subset(carDataFinal, splitLR == TRUE)
testDataLR<- subset(carDataFinal, splitLR == FALSE)
nrow(trainDataLR)
nrow(testDataLR)
prop.table(table(trainDataLR$Uses_Car))
#Train Data for Bagging
trainDataBag=trainDataLR
#Test Data for Bagging
testDataBag=testDataLR
#Train Data for Boosting
trainDataBoost=trainDataLR
#Test Data for Boosting
testDataBoost=testDataLR
#Test Data for Naive Bayes
testDataNB=testDataLR
#Test Data for KNN
testDataKNN=testDataLR
#Percentage positive in Train
table(trainDataLR$Uses_Car)
sum(trainDataLR$Uses_Car == 1)/nrow(trainDataLR)
#Percentage positive in test
table(testDataLR$Uses_Car)
sum(testDataLR$Uses_Car == 1)/nrow(testDataLR)
#Data is imbalanced so we SMOTE
carsTrainSMOTE<-SMOTE(Uses_Car~.,trainDataLR,perc.over=300,perc.under=200)                    0)
prop.table(table(carsTrainSMOTE$Uses_Car))
dim(trainDataLR)
dim(carsTrainSMOTE)



#####################Logistic Regression######################################
set.seed(seed)
#Logistic: Model2
LRModel=glm(Uses_Car~., data = carsTrainSMOTE, family = binomial)
##Check AIC and statistical significant variables
summary(LRModel)
#Check Loglikelihood and p value
lrtest(LRModel)

#Check multicollienearity
vif(LRModel)
dim(carsTrainSMOTE)
# Predicting test set
testDataLR$response=predict(LRModel,newdata = testDataLR[,-8],type="response")
testDataLR$Usage_Predict=ifelse(testDataLR$response<.5,"0","1")
#Convert to factor
testDataLR$Usage_Predict=as.factor(testDataLR$Usage_Predict)
tabTestLR=table(testDataLR$Usage_Predict,testDataLR$Uses_Car)
tabTestLR
#Confusion Matrix
confusionMatrix(testDataLR$Usage_Predict,testDataLR$Uses_Car, positive="1")
#AUC KS GINI
ROCRTestLR=prediction(testDataLR$response,testDataLR$Uses_Car)
aucTestLR=as.numeric(performance(ROCRTestLR,"auc")@y.values)
print(paste('Area Under the Curve for test Dataset:',aucTestLR))
perfTestLR=performance(ROCRTestLR,"tpr","fpr")
plot(perfTestLR,main="AUC ROC Curve for test dataset")
KSTestLR <- max(attr(perfTestLR, 'y.values')[[1]]-attr(perfTestLR, 'x.values'
)[[1]])
print(paste('K-S Value for test Dataset',KSTestLR))
giniTestLR = ineq(testDataLR$response, type="Gini")
print(paste('Gini Coefficient for test dataset:',giniTestLR))

###################KNN###############################
#For KNN we convert Gender in 0 and 1
set.seed(seed)
carDataknn=carDataFinal
carDataknn$Gender=ifelse(carDataknn$Gender=="Male",1,0)
carDataknn$Gender=as.factor(carDataknn$Gender)
str(carDataknn)
#normalizing the data
norm = function(x) { (x- min(x))/(max(x) - min(x)) }
normKNN = as.data.frame(lapply(carDataknn[,-c(2,3,4,7,8)], norm))
head(normKNN)
finalDataKNN = cbind(carDataknn[,c(2,3,4,7,8)], normKNN)
head(finalDataKNN)
# Data partitioning
splitKNN = sample.split(finalDataKNN$Uses_Car, SplitRatio = 0.7)
trainDataKnn = subset(finalDataKNN, splitKNN == T)
testDataKnn = subset(finalDataKNN, splitKNN == F)
head(trainDataKnn)
head(testDataKnn)
knnTrainSMOTE<-SMOTE(Uses_Car~., trainDataKnn,perc.over = 300,perc.under = 200)
prop.table(table(knnTrainSMOTE$Uses_Car))
str(knnTrainSMOTE)
predKnn9 = knn(knnTrainSMOTE[-5], testDataKnn[-5], knnTrainSMOTE[,5], k = 9)
tabKnn9 = table(testDataKnn[,5], predKnn9)
tabKnn9
accKnn9=sum(diag(tabKnn9)/sum(tabKnn9))
print(paste('Accuracy for KNN with k=9:',accKnn9))
#Confusion Matrix
confusionMatrix(predKnn9,testDataKnn$Uses_Car, positive="1")




set.seed(seed)
#################NB Model building on train data#############################
ModelNB = naiveBayes( carsTrainSMOTE$Uses_Car~., data = carsTrainSMOTE)
ModelNB
#Model on test data
testDataNB$CarUsage_Predict=predict(ModelNB, newdata = testDataNB, type = "class")
testTabNB=table(testDataNB$CarUsage_Predict,testDataNB$Uses_Car)
testTabNB
#Confusion Matrix
confusionMatrix(testDataNB$CarUsage_Predict,testDataNB$Uses_Car, positive="1")


###################Bagging#####################################################
set.seed(seed)
ModelBagging<- bagging(trainDataBag$Uses_Car ~.,
                       data=trainDataBag,
                       control=rpart.control(maxdepth=5, minsplit=15))
summary(ModelBagging)
#training
trainDataBag$pred.class <- predict(ModelBagging, trainDataBag)
table(trainDataBag$Uses_Car,trainDataBag$pred.class)
confusionMatrix(trainDataBag$pred.class,trainDataBag$Uses_Car, positive="1")

#Testing
testDataBag$pred.class <- predict(ModelBagging, testDataBag)
table(testDataBag$Uses_Car,testDataBag$pred.class)
confusionMatrix(testDataBag$pred.class,testDataBag$Uses_Car, positive="1")


#################Boosting######################################################
set.seed(seed)
gbm.fit <- gbm(formula = Uses_Car ~ ., distribution = "multinomial", 
               data = trainDataBoost,n.trees = 10000, interaction.depth = 1, 
               shrinkage = 0.001,cv.folds = 5,n.cores = NULL,verbose = FALSE)
gbm.fit
#Training
trainDataBoost$response= predict(gbm.fit, trainDataBoost[-8], type = "response")
trainDataBoost$pred.CarUsage <- apply(trainDataBoost$response, 1, which.max)
#Get predicted values in 0 and 1
trainDataBoost$CarUsage_Predict<-ifelse(trainDataBoost$pred.CarUsage==1,0,1)
#Convert to fator
trainDataBoost$CarUsage_Predict=as.factor((trainDataBoost$CarUsage_Predict))
table(trainDataBoost$Uses_Car,trainDataBoost$CarUsage_Predict)
confusionMatrix(trainDataBoost$CarUsage_Predict,trainDataBoost$Uses_Car, positive="1")


#Testing
testDataBoost$response= predict(gbm.fit, testDataBoost[-8], type = "response")
testDataBoost$pred.CarUsage <- apply(testDataBoost$response, 1, which.max)
#Get predicted values in 0 and 1
testDataBoost$CarUsage_Predict<-ifelse(testDataBoost$pred.CarUsage==1,0,1)
#Convert to fator
testDataBoost$CarUsage_Predict=as.factor((testDataBoost$CarUsage_Predict))
table(testDataBoost$Uses_Car,testDataBoost$CarUsage_Predict)
confusionMatrix(testDataBoost$CarUsage_Predict,testDataBoost$Uses_Car, positive="1")


vip::vip(gbm.fit, num_features = 8, bar = FALSE)








