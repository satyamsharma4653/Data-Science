rm(list = ls(all=T))

library(ggplot2)
library(caret)
library(ggcorrplot)
library(ROSE)
library(randomForest)
library(rpart)
library(caret)
library(e1071)

setwd("C:/Users/Chinmay/Documents/Santander Customer Prediction")
#getwd()

#Reading Train and Test data
my_train = read.csv('train.csv',header = T)
View(head(my_train))
my_test = read.csv('test.csv' , header = T)

#Checking Dimensions of train and test dataframes
dim(my_train)
dim(my_test)

#Descriptive Statistics
summary(my_train)
summary(my_test)

#Observation: In every column mean and median are almost same--------------------

my_train_ID = my_train$ID_code
my_test_ID = my_test$ID_code

#Removing ID_code from original dataset
my_train$ID_code = NULL
my_test$ID_code = NULL

dim(my_train)
dim(my_test)

###############################################################################
################          DATA PRE PROCESSING           #######################
###############################################################################

#--------------------------MISSING VALUE ANALYSIS------------------------------


calculateMissingValues = function(df){
  missing_val = data.frame(apply(df , 2 , function(x){sum(is.na(x))}))
  missing_val$Columns = row.names(missing_val)
  names(missing_val)[1] =  "Missing_percentage"
  missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(df)) * 100
  missing_val = missing_val[order(-missing_val$Missing_percentage),]
  row.names(missing_val) = NULL
  missing_val = missing_val[c(2,1)]
  
  barplot(missing_val$Missing_percentage , xlab = "Feature Variables" ,
          ylab = "Amount of Missing Value Detected",
          main = "Missing Value Visualization")
  return(missing_val)
}

#checking missing values of train data and test data
calculateMissingValues(my_train)
calculateMissingValues(my_train)

#Observation: No missing values were detected

##################################################  OUTLIER ANALYSIS  ########################
#boxplot(my_train$var_100)

check_Outliers = function(df){
  jpeg("Boxplot_var_0-var_20.jpg", width = 1200 , height = 800)
  boxplot(df[,1:21], col = "green" ,outline = T)
  dev.off()
  
  jpeg("Boxplot_var_21-var_40.jpg", width = 1200 , height = 800)
  boxplot(df[,22:41], col = "green" ,outline = T)
  dev.off()
  
  jpeg("Boxplot_var_41-var_60.jpg", width = 1200 , height = 800)
  boxplot(df[,42:61], col = "green" ,outline = T)
  dev.off()
  
  jpeg("Boxplot_var_61-var_80.jpg", width = 1200 , height = 800)
  boxplot(df[,62:81], col = "green" ,outline = T)
  dev.off()
  
  jpeg("Boxplot_var_81-var_100.jpg", width = 1200 , height = 800)
  boxplot(df[,82:101], col = "green" ,outline = T)
  dev.off()
  
  jpeg("Boxplot_var_101-var_120.jpg", width = 1200 , height = 800)
  boxplot(df[,102:121], col = "green" ,outline = T)
  dev.off()
  
  jpeg("Boxplot_var_121-var_140.jpg", width = 1200 , height = 800)
  boxplot(df[,122:141], col = "green" ,outline = T)
  dev.off()
  
  jpeg("Boxplot_var_141-var_160.jpg", width = 1200 , height = 800)
  boxplot(df[,142:161], col = "green" ,outline = T)
  dev.off()
  
  jpeg("Boxplot_var_161-var_180.jpg", width = 1200 , height = 800)
  boxplot(df[,162:181], col = "green" ,outline = T)
  dev.off()
  
  jpeg("Boxplot_var_181-var_200.jpg", width = 1200 , height = 800)
  boxplot(df[,182:201], col = "green" ,outline = T)
  dev.off()
  
}

#Visualizing Outliers using Boxplots
check_Outliers(my_train)

table(my_train$target)
nrow(my_train)
prop.table(table(my_train$target))

#Fetching column names of feature variables
cnames = colnames(my_train[,2:201])

#Function for removing outlier rows
remove_Outliers = function(my_df){
  
  #loop to remove outliers from all variables
  for(i in cnames){
    print(i)
    val = my_df[,i][my_df[,i] %in% boxplot.stats(my_df[,i])$out]    #selecting rows having outliers
    print(length(val))
    my_df = my_df[which(!my_df[,i] %in% val),]
  }
  return(my_df)
}

#Function to replace outliers with NA
replace_Outliers = function(my_df){
  #Replace all outliers with NA and impute
  #create NA on "custAge
  for(i in cnames){
     val = my_df[,i][my_df[,i] %in% boxplot.stats(my_df[,i])$out]
     #print(length(val))
     my_df[,i][my_df[,i] %in% val] = NA
  }
  return(my_df)
}

#------------------------Type 1 Outlier Handling(dropping rows)------------------------
cleaned_df = remove_Outliers(my_train)    #in this dataframe rows having outliers have been removed

#Checking dimension of dataset after removing outlier rows
dim(cleaned_df)

#Calculating percentage of data that have been removed
(nrow(my_train) - nrow(cleaned_df))/nrow(my_train)   
#Observation: Since only 12.4% of data have been removed we can consider dropping rows having outliers

#-----------------------Type 2 Outlier Handling(replacing with NA)----------------------------
imputed_df = replace_Outliers(my_train)   #here we have replaced outlier values with NA

#Calculating Missing Values after replacing outliers with NA
calculateMissingValues(imputed_df)

#imputing missing values with their mean values
impute_NA = function(df){
  for(i in cnames)
  {
    df[is.na(df[,i]), i] <- mean(df[,i], na.rm = TRUE)
  }
  return(df)
}

imputed_df = impute_NA(imputed_df)
print("No. of Na's after imputing them: ")
print(sum(is.na(imputed_df)))

#checking count of target variable
table(cleaned_df$target)
barplot(prop.table(table(cleaned_df$target)),
        col = rainbow(2),
        main = "Target Class Distribution in Cleaned DF")

table(imputed_df$target)
barplot(prop.table(table(imputed_df$target)),
        col = rainbow(2),
        main = "Target Class Distribution in Imputed Outlier DF")

###################################CORRELATION ANALYSIS##########################
library(ggcorrplot)

ggcorrplot(cor(cleaned_df), tl.cex = 0.5 ,
  title = "Correlattion Analysis",
  colors = c('blue','black','white'))


ggcorrplot(cor(imputed_df),tl.cex = 0.5 ,
           title = "Correlattion Analysis",
           colors = c('blue','black','white'))

#Observation: All the variables are independent

###########################################FEATURE SCALING###########################
#-----------------------------------------Standardization----------------------------

standardize_df = function(my_df)
{
  for(i in cnames)
  {
    print(i)
    my_df[,i] = (my_df[,i] - mean(my_df[,i]))/sd(my_df[,i])
  }
  return(my_df)
}

cleaned_df = standardize_df(cleaned_df)
imputed_df = standardize_df(imputed_df)


#-------------------------CREATING TRAIN TEST SPLIT-----------------------------

#Outliers have been removed in this train test split
set.seed(123)
ind <- sample(2 , nrow(cleaned_df), replace = TRUE , prob = c(0.7,0.3))
cs_train = cleaned_df[ind ==1 ,]    
cs_test = cleaned_df[ind ==2 ,]



#Outliers have been imputed in this train test split
set.seed(123)
ind <- sample(2 , nrow(imputed_df), replace = TRUE , prob = c(0.7,0.3))
is_train = imputed_df[ind ==1 ,]    
is_test = imputed_df[ind ==2 ,]


##########################################     MODELLING    ##############################

#Function to calculate accuracy of model
calculate_accuracy = function(c_matrix)
{
  tn =c_matrix[1,1]
  tp =c_matrix[2,2]
  fp =c_matrix[1,2]
  fn =c_matrix[2,1]
  p =round((tp)/(tp+fp),2)
  r =round((tp)/(tp+fn),2)
  a = round(((tp+tn)/(tp+tn+fp+fn))*100,2)
  f1=2*((p*r)/(p+r))
  fpr = round((fp)/(fp+tn),2)
  fnr = round((fn)/(fn+tp),2)
  metrics = c("Accuracy","Precision:","Recall:","False Positive Rate:","False Negative Rate","F1 Score:")
  values = c(a,p,r,fpr,fnr,f1)
  print(data.frame(metrics,values))
}

#----------------------------------------LOGISTIC REGRESSION------------------------
#MODEL 1 - Cleaned DF NORMAL--------------------------########################################
lr_model_1 = glm(target~., data = cs_train, family = "binomial")
#summary(lr_model_1)
y_prob = predict(lr_model_1, newdata = cs_test)
y_pred = ifelse(y_prob > 0.5 , 1 , 0)

conf_matrix = table(cs_test$target , y_pred)
conf_matrix
calculate_accuracy(conf_matrix)
dim(cs_test)

library(pROC)
roc=roc(cs_test$target, y_prob)
print(roc)
plot(roc ,main ="Logistic Regression base Roc ")


#1             Accuracy 91.3800000
#2           Precision:  0.7800000
#3              Recall:  0.1700000
#4 False Positive Rate:  0.0100000
#5  False Negative Rate  0.8300000
#6            F1 Score:  0.2791579
#7                 AUC:  0.8599

#Clearing Space
rm(y_prob)
rm(y_pred)
rm(conf_matrix)
rm(roc)


#MODEL 2 - Cleaned DF - OVERSAMPLED------------------------#########################################


over_cs = ovun.sample(target~. , data = cs_train , method = "over" , N = 200000)$data
lr_model_2 = glm(target~., data = over_cs, family = "binomial")
y_prob = predict(lr_model_2, newdata = cs_test)
y_pred = ifelse(y_prob > 0.5 , 1 , 0)

conf_matrix = table(cs_test$target , y_pred)
conf_matrix
confusionMatrix(conf_matrix)
calculate_accuracy(conf_matrix)



roc=roc(cs_test$target, y_prob)
print(roc)
plot(roc ,main ="Logistic Regression base Roc ")

#1             Accuracy 86.8900
#2           Precision:  0.3900
#3              Recall:  0.6100
#4 False Positive Rate:  0.1000
#5  False Negative Rate  0.3900
#6            F1 Score:  0.4758
#7                 AUC:  0.8595

#Clearing Space
rm(y_prob)
rm(y_pred)
rm(conf_matrix)
rm(roc)

#MODEL 3 - Imputed DF Normal--------------------------###########################################

lr_model_3 = glm(target~., data = is_train, family = "binomial")

y_prob = predict(lr_model_3, newdata = cs_test)
y_pred = ifelse(y_prob > 0.5 , 1 , 0)

conf_matrix = table(cs_test$target , y_pred)
conf_matrix
confusionMatrix(conf_matrix)
calculate_accuracy(conf_matrix)



roc=roc(cs_test$target, y_prob)
print(roc)
plot(roc ,main ="Logistic Regression Model 3")

#1             Accuracy 91.44000000
#2           Precision:  0.76000000
#3              Recall:  0.01000000
#4 False Positive Rate:  0.01000000
#5  False Negative Rate  0.82000000
#6            F1 Score:  0.01974026
#7                 AUC:  0.8615

#Clearing Space
rm(y_prob)
rm(y_pred)
rm(conf_matrix)
rm(roc)


#MODEL 4 - Imputed DF Oversampled----------------------###########################################

over_is = ovun.sample(target~. , data = is_train , method = "over" , N = 200000)$data
lr_model_4 = glm(target~., data = over_is, family = "binomial")

y_prob = predict(lr_model_4, newdata = cs_test)
y_pred4 = ifelse(y_prob > 0.5 , 1 , 0)

conf_matrix = table(cs_test$target , y_pred4)
conf_matrix
confusionMatrix(conf_matrix)
calculate_accuracy(conf_matrix)



roc=roc(cs_test$target, y_prob)
print(roc)
plot(roc ,main ="Logistic Regression model 4 ")

#1             Accuracy 88.9500000
#2           Precision:  0.4500000
#3              Recall:  0.0700000
#4 False Positive Rate:  0.0700000
#5  False Negative Rate  0.4700000
#6            F1 Score:  0.1211538
#7                 AUC:  0.8615

#Clearing Space
rm(y_prob)
rm(y_pred)
rm(conf_matrix)
rm(roc)


#Observation: Out of all the 4 models of Logistic Regression we will keep the Model 2, because it has more F1 score
#in comparasion to other models. Also Model 2 is having high recall.
#It is also observed that the model in which outliers have been removed is performing well.

rm(lr_model_1)
rm(lr_model_3)
rm(lr_model_4)
rm(imputed_df)
rm(is_train)
rm(is_test)
rm(over_is)


final_train = over_cs  #finalized over sampled data for all other models
final_test = cs_test


#MODEL 5 DECISION TREE - OVERSAMPLED CLEANED DATA----------------------------------##################################
dt_model = rpart( target~. ,data =final_train )

y_prob = predict(dt_model, newdata = final_test)
y_pred = ifelse(y_prob > 0.5 , 1 , 0)

conf_matrix = table(final_test$target , y_pred)
conf_matrix
confusionMatrix(conf_matrix)
calculate_accuracy(conf_matrix)


library(pROC)
roc=roc(final_test$target, y_prob)
print(roc)
plot(roc ,main ="Decision Tree Model 1 ROC ")

#1             Accuracy 61.9400000
#2           Precision:  0.1300000
#3              Recall:  0.5300000
#4 False Positive Rate:  0.3700000
#5  False Negative Rate  0.4700000
#6            F1 Score:  0.2087879
#7                 AUC:  0.5881

#Clearing Space
rm(y_prob)
rm(y_pred)
rm(conf_matrix)
rm(roc)
#rm(dt_model_2)

#MODEL 6 DECISION TREE - TUNED --------------------------------------
#rpart control variable 

dt_model_2 = rpart( target~. ,data =final_train , control = rpart.control(cp = 0.02,minsplit = 5,maxdepth = 5))

y_prob6 = predict(dt_model_2, newdata = final_test)
y_pred6 = ifelse(y_prob6 > 0.5 , 1 , 0)

conf_matrix = table(final_test$target , y_pred6)
conf_matrix
confusionMatrix(conf_matrix)
calculate_accuracy(conf_matrix)



roc=roc(final_test$target, y_prob6)
print(roc)
plot(roc ,main ="Decision Tree Model 2 ROC ")

#1             Accuracy 80.6900000
#2           Precision:  0.1600000
#3              Recall:  0.2300000
#4 False Positive Rate:  0.1300000
#5  False Negative Rate  0.7700000
#6            F1 Score:  0.1887179
#7                 AUC:  0.5498


#MODEL 7 Naive Bayes-----------------------------------
table(cs_train$target)
prop.table(table(cs_train$target))

nb_model  =naiveBayes(target~.  , data =cs_train)  


y_prob =predict(nb_model , final_test  ,type='raw')
y_pred = ifelse(y_prob[,2] >0.5, 1, 0)


conf_matrix= table(final_test$target , y_pred)
conf_matrix

calculate_accuracy(conf_matrix)

roc=roc(final_test$target, y_prob[,2] )
print(roc)
plot(roc ,main="Model 1 Naive Bayes")

#1             Accuracy 92.2400000
#2           Precision:  0.7100000
#3              Recall:  0.3500000
#4 False Positive Rate:  0.0200000
#5  False Negative Rate  0.6500000
#6            F1 Score:  0.4688679
#7                 AUC:  0.89

rm(nb_model)

#MODEL 8 NAIVE BAYES WITH OVER SAMPLED DATA-------------------------------------------------#########################
nb_model_2  =naiveBayes(target~.  , data =final_train)  


y_prob = predict(nb_model_2 , final_test  ,type='raw')
y_pred = ifelse(y_prob[,2] >0.5, 1, 0)


conf_matrix= table(final_test$target , y_pred)
conf_matrix

calculate_accuracy(conf_matrix)

roc=roc(final_test$target, y_prob[,2] )
print(roc)
plot(roc ,main="Model 2 Naive Bayes")

#1             Accuracy 83.8300000
#2           Precision:  0.3500000
#3              Recall:  0.7600000
#4 False Positive Rate:  0.1500000
#5  False Negative Rate  0.2400000
#6            F1 Score:  0.4792793
#7                 AUC:  0.8889


