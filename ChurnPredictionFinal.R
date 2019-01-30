dataset <- read.table(file.choose(), header=TRUE, sep=",") 
dataset$churned <- as.factor(dataset$churned)
sum(is.na(dataset)) #5 missing values in the dataset
sum(is.na(dataset$continent)) #All in continent
dataset <- na.omit(dataset) #Erase missing values 
View(dataset)
str(dataset)

install.packages('caret')
library(caret) 
require(DMwR) #for computing SMOTE
require(pROC) #for computing AUC
install.packages('RWeka')
library(RWeka) #for J48
install.packages("partykit")
library(partykit)
library(randomForest)


#Split data 70:30
set.seed(1)
splitIndex <- createDataPartition(dataset$churned, p = .70, list = FALSE, times = 1)
train <- dataset[ splitIndex,]
test <- dataset[-splitIndex,]
str(train)
str(test)

#################

#Create the initial logistic regression model
model.glm.initial <- glm(churned~.,family = 'binomial',data=train)
summary(model.glm.initial)

#Predict accuracy of the intial model
pred.initial.glm = predict(model.glm.initial, test, type="response")
classificationtable.initial.glm <- table(test$churned,pred.initial.glm>0.5)
print(classificationtable.initial.glm)
accuracy.initial.glm <- (178+2993)/(178+84+297+2993)
print(accuracy.initial.glm)

#Use trainControl for CV
train.control.glm<- trainControl(method="cv", number=10)

#Cross-validated glm:
set.seed(1)
model.glm.cv <- train(churned~., 
                    data=train, trControl=train.control.glm, method="glm")

#Predict the accuracy of the cv model
pred.glm <- predict(model.glm.cv, newdata = test)
classificationtable.glm <- confusionMatrix(test$churned,pred.glm) 
print(classificationtable.glm)  
accuracy.glm <- (178+2993)/(178+84+297+2993) #accuracy did not change
print(accuracy.glm)

#Compute AUC for glm
pred.glm <- as.numeric(pred.glm) #otherwise roc won't work
auc.glm <- roc(test$churned, pred.glm)
print(auc.glm)

#################

#Model the decision tree (J48) - J48 is pruned by default 
set.seed(1)
model.J48 <- J48(churned~., data = train)

#Get the classification table and accuracy of the tree
pred.J48 <- predict(model.J48, test[,-9], type='class')
classificationtable.J48 <- table(pred.J48,test[,9])
print(classificationtable.J48)
accuracy.J48 <- sum(diag(classificationtable.J48))/sum(classificationtable.J48)
print(accuracy.J48)

#Compute AUC
pred.J48 <- as.numeric(pred.J48)
auc.J48 <- roc(test$churned,pred.J48)
print(auc.J48)

####################

#Model the random forest
set.seed(1)
model.rf <- randomForest(churned~.,train, ntree = 500, mtry = 4, 
                         importance = TRUE)

#check varible importance & visualize it
importance(model.rf) 
varImpPlot(model.rf) 

#Get the classification table and accuracy of the intial model
pred.rf <- predict(model.rf, test[,-9], type='class')
pred.rf <- as.numeric(pred.rf)
classificationtable.rf <- table(pred.rf,test[,9])
print(classificationtable.rf) 
accuracy.rf <- sum(diag(classificationtable.rf))/sum(classificationtable.rf)
print(accuracy.rf)

#Compute AUC
auc.rf <- roc(test$churned,pred.rf)
print(auc.rf)


################

#time to SMOTE it!!!
set.seed(1)
trainSMOTE <- SMOTE(churned ~ ., train, perc.over = 100, perc.under=200)
print(prop.table(table(trainSMOTE$churned))) #ratio of 1s & 0s


#Model SMOTEd logistic regression and measure performance
set.seed(1)
model.glmSMOTE <- train(churned ~ ., 
                        data = trainSMOTE, method = "glm", trControl = ctrl)
pred.glmSMOTE <- predict(model.glmSMOTE, test[,-9])
classificationtable.glmSMOTE <- table(pred.glmSMOTE,test[,9])
print(classificationtable.glmSMOTE) 
accuracy.glmSMOTE <- 
  sum(diag(classificationtable.glmSMOTE))/sum(classificationtable.glmSMOTE)
print(accuracy.glmSMOTE)

pred.glmSMOTE <- as.numeric(pred.glmSMOTE)
auc.glmSMOTE <- roc(test$churned,pred.glmSMOTE)
print(auc.glmSMOTE)

#Model SMOTEd decision tree and measure performance
set.seed(1)
model.J48SMOTE <- train(churned ~ ., data = trainSMOTE, 
                        method = "J48", trControl = ctrl)
pred.J48SMOTE <- predict(model.J48SMOTE, test[,-9])
classificationtable.J48SMOTE <- table(pred.J48SMOTE,test[,9])
print(classificationtable.J48SMOTE) 
accuracy.J48SMOTE <- 
  sum(diag(classificationtable.J48SMOTE))/sum(classificationtable.J48SMOTE)
print(accuracy.J48SMOTE)

pred.J48SMOTE <- as.numeric(pred.J48SMOTE)
auc.J48SMOTE <- roc(test$churned,pred.J48SMOTE)
print(auc.J48SMOTE)


#Model SMOTEd random forest and measure performance
set.seed(1)
model.rfSMOTE <- train(churned ~ ., 
                       data = trainSMOTE, method = "rf", trControl = ctrl)
pred.rfSMOTE <- predict(model.rfSMOTE, test[,-9])
classificationtable.rfSMOTE <- table(pred.rfSMOTE,test[,9])
print(classificationtable.rfSMOTE) 
accuracy.rfSMOTE <- 
  sum(diag(classificationtable.rfSMOTE))/sum(classificationtable.rfSMOTE)
print(accuracy.rfSMOTE)

pred.rfSMOTE <- as.numeric(pred.rfSMOTE)
auc.rfSMOTE <- roc(test$churned,pred.rfSMOTE)
print(auc.rfSMOTE)


###########

#Summarize the all the models

#accuracy
accuracy.values <- matrix(c(accuracy.glm,accuracy.J48,
                      accuracy.rf,accuracy.glmSMOTE,
                      accuracy.J48SMOTE,accuracy.rfSMOTE))
accuracy.names <- matrix(c('glm','J48','rf','SMOTEd glm',
                           'SMOTEd J48','SMOTEd rf'))
accuracy.table <- cbind(accuracy.names,accuracy.values)
colnames(accuracy.table) <- c('model','accuracy')
accuracy.table <- as.table(accuracy.table)
print(accuracy.table)

#AUC
auc.values <- matrix(c(auc.glm$auc,auc.J48$auc,auc.rf$auc,
                       auc.glmSMOTE$auc, auc.J48SMOTE$auc, 
                       auc.rfSMOTE$auc))
auc.names <- accuracy.names
auc.table <- cbind(auc.names, auc.values)
colnames(auc.table) <- c('model','AUC')
auc.table <- as.table(auc.table)
print(auc.table)


#Plot all ROC curves

rf.plot <- plot(auc.rf, print.auc=TRUE, col = '#00CC66', 
     main="ROC curves comparison",
     legacy.axes=TRUE, lwd = 1.5, lty = 'dotted')

rfSMOTE.plot <- plot(auc.rfSMOTE, add = TRUE, print.auc=TRUE, 
                     col = '#00CC66', legacy.axes=TRUE, 
                     print.auc.y = 0.5, print.auc.x = 0.25, lwd = 1.5)
J48.plot <- plot(auc.J48, add = TRUE, col = '#330099', 
     legacy.axes=TRUE, print.auc = TRUE, 
     print.auc.y = 0.4, lwd = 1.5, lty = 'dotted')

J48SMOTE.plot <- plot(auc.J48SMOTE, add = TRUE, col = '#330099', 
                      legacy.axes=TRUE, print.auc = TRUE, 
                      main="ROC curves comparison",
                      print.auc.y = 0.4, print.auc.x = 0.25, lwd = 1.5)

glm.plot <- plot(auc.glm, add = TRUE, col = '#CC33CC', 
     legacy.axes=TRUE, print.auc = TRUE, 
     print.auc.y = 0.3, lwd = 1.5, lty = 'dotted')

glmSMOTE.plot <- plot(auc.glmSMOTE, add = TRUE, col = '#CC33CC', 
     legacy.axes=TRUE, print.auc = TRUE, 
     print.auc.y = 0.3, print.auc.x = 0.25, lwd = 1.5)



###################

#Measure the improtance of predictors - information gain ratio (Chen et al. 2014)
install.packages('FSelector')
library(FSelector)
information.gain(churned~., data = dataset)
#clearly indicates  avg_cycles_count & pricing_plan_id are 
#the most important predictors
