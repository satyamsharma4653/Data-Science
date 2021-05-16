rm(list = ls(all=T))

library(dplyr)
library(ggplot2)
library(caret)
library(ggcorrplot)
library(ROSE)
library(randomForest)
library(rpart)
library(caret)
library(e1071)
library(ggplot2)
library(tidyr)
library(purrr)
library(ggcorrplot)   #for Correlation plot
library(cluster)      #for plotting clusters
library(factoextra)   #for creating graphs of clusters


setwd("C:/Users/Chinmay/Documents/Credit Card Segmentation/")
getwd()

#Reading File
my_train = read.csv("credit-card-data.csv")
#View(my_train)

#Checking Dimensions of train and test dataframes
dim(my_train)

#Descriptive Statistics
summary(my_train)


my_train_ID = my_train$CUST_ID

#Removing CUST ID from training data
my_train$CUST_ID = NULL


#--------------------------------------------MISSING VALUE ANALYSIS-------------------------------------------------
missing_val = apply(my_train, 2, function(x){sum(is.na(x))})
missing_val = data.frame(missing_val)


#MINIMUM_PAYMENTS AND CREDIT_LIMIT are having missing values so Imputing them with MEDIAN METHOD
my_train$MINIMUM_PAYMENTS[is.na(my_train$MINIMUM_PAYMENTS)] = median(my_train$MINIMUM_PAYMENTS , na.rm = T)
my_train$CREDIT_LIMIT[is.na(my_train$CREDIT_LIMIT)] = median(my_train$CREDIT_LIMIT , na.rm = T)




#CREATING NEW KPI's
my_train$MONTHLY_AVG_PURCHASE = my_train$PURCHASES / my_train$TENURE
my_train$MONTHLY_AVG_CASH_ADVANCE = my_train$CASH_ADVANCE / my_train$TENURE
my_train$PTMP = my_train$PAYMENTS / my_train$MINIMUM_PAYMENTS
my_train$LIMIT_USAGE = my_train$BALANCE / my_train$CREDIT_LIMIT
my_train$PURCHASE_TYPE = 0
my_train[my_train$ONEOFF_PURCHASES ==0 & my_train$INSTALLMENTS_PURCHASES ==0 ,'PURCHASE_TYPE'] = 'none'
my_train[my_train$ONEOFF_PURCHASES >0 & my_train$INSTALLMENTS_PURCHASES ==0 ,'PURCHASE_TYPE'] = 'oneoff'
my_train[my_train$ONEOFF_PURCHASES ==0 & my_train$INSTALLMENTS_PURCHASES >0 ,'PURCHASE_TYPE'] = 'installment'
my_train[my_train$ONEOFF_PURCHASES >0 & my_train$INSTALLMENTS_PURCHASES >0 ,'PURCHASE_TYPE'] = 'both'
my_train$both = 0
my_train$oneoff = 0
my_train$installment = 0 
my_train$none = 0
my_train[my_train$PURCHASE_TYPE == 'both','both']=1
my_train[my_train$PURCHASE_TYPE == 'oneoff','oneoff']=1
my_train[my_train$PURCHASE_TYPE == 'installment','installment']=1
my_train[my_train$PURCHASE_TYPE == 'none','none']=1

#my_train$PURCHASE_TYPE = as.factor(my_train$PURCHASE_TYPE)


my_train$PURCHASE_TYPE = NULL


cnames = colnames(my_train)
cnames

str(my_train)

#---------------------------------------------STANDARDIZATION--------------------------------------------------------
#for(i in cnames){
#  #print(i)
#  my_train[,i] = (my_train[,i] - mean(my_train[,i]))/sd(my_train[,i])
#}

#--------------------------------------------Checking for outliers-------------------------------------------------
boxplot(my_train)

#as we can see there are lot of outliers, so we can't remove them
#SO we'll impute them using log transformations



ggplot(data = my_train, aes(BALANCE)) + geom_histogram()


#making a copy of actual dataset
copy_df = my_train

#my_train %>% keep(is.numeric) %>% gather() %>% head()

#Checking for outliers by looking at distributions
my_train %>%
  keep(is.numeric) %>%                     # Keep only numeric columns
  gather() %>%                             # Convert to key-value pairs
  ggplot(aes(value)) +                     # Plot the values
  facet_wrap(~ key, scales = "free") +   # In separate panels
  geom_density()

#For checking skewness
skewness(my_train$BALANCE)

pskewed_data = c('BALANCE','CASH_ADVANCE','CASH_ADVANCE_FREQUENCY',
                'CASH_ADVANCE_TRX','CREDIT_LIMIT',
                'INSTALLMENT_PURCHASES','MINIMUM_PAYMENTS',
                'ONEOFF_PURCHASES','ONEOFF_PURCHASES_FREQUENCY',
                'PAYMENTS','PRC_FULL_PAYMENT','PURCHASES',
                'PURCHASES_INSTALLMENTS_FREQUENCY','PURCHASES_TRX')



# we observed that most of the data is positive skewed so we'll apply log(x+1) transformation


#Applying log(x+1) transformation to remove outlier effect
my_train = apply(my_train, 2, function(x){log(x+1)})


#Checking Range of each column
for(i in colnames(my_train)){
  print(i)
  print(range(my_train[i]))
  
}

#Normalization
for(i in colnames(my_train)){
  print(i)
  my_train[,i] = (my_train[,i] - min(my_train[,i]))/
    (max(my_train[,i] - min(my_train[,i])))
}

###################################CORRELATION ANALYSIS##########################
library(ggcorrplot)

ggcorrplot(cor(my_train), tl.cex = 0.5 ,
           title = "Correlattion Analysis",
           colors = c('blue','black','white'))+ theme(axis.text.x = element_text(face="bold", color="#993333", 
                                                                                 size=9, angle=45),
                                                      axis.text.y = element_text(face="bold", color="#993333", 
                                                                                 size=9, angle=0))
#View(cor(my_train))

#---------------------------DIMENSIONALITY REDUCTION USING PCA--------------------------



pca_train <- my_train
pca = prcomp(pca_train,scale. = T)

loadings <- as.data.frame(pca$x)
#View(loadings)

Matrix <- pca$rotation

std_dev <- pca$sdev
pr_comp_var <- std_dev^2
round(pr_comp_var,5)


prop_var_ex <- pr_comp_var/sum(pr_comp_var)
round(prop_var_ex,4)

plot(cumsum(prop_var_ex), xlab = "Principal Component",ylab = "Proportion of Variance Explained",type = "b")


pca_df = loadings[1:2]
#View(pca_df)


#-----------------------------MODELLING-----------------------------
kmeans(pca_df, 4, iter.max = 10, nstart = 1)

#install.packages('factoextra')
library(factoextra)   #for creating graphs of clusters

fviz_nbclust(pca_df,kmeans , method = 'wss')+
  geom_vline(xintercept = 3, linetype = 2)

fviz_nbclust(pca_df, kmeans, method = "silhouette")

set.seed(123)
km.res <- kmeans(pca_df, 3, iter.max = 10, nstart = 1)
km.res


dd <- cbind(pca_df, cluster = km.res$cluster)
head(dd)

cl1 = as.integer(rownames(dd[dd['cluster']==1,]))
head(cl1)
cl2 = as.integer(rownames(dd[dd['cluster']==2,]))
cl3 = as.integer(rownames(dd[dd['cluster']==3,]))
cl4 = as.integer(rownames(dd[dd['cluster']==4,]))


cluster1 = my_train[cl1,]
cluster2 = my_train[cl2,]
cluster3 = my_train[cl3,]
cluster4 = my_train[cl4,]


fviz_cluster(km.res, pca_df,
             palette = "Set2", ggtheme = theme_minimal())


#--------------------HIERARCHICAL CLUSTERING------------------------
#using dendogram to find optimal no. of clusters

dendogram = hclust(dist(pca_df, method = 'euclidean') , method = 'ward.D')
plot(dendogram,
     main = paste('Dendogram'),
     xlab = 'pca1',
     tlab = 'pca2')


#fitting hierarchical clusteting to dataset
hc = hclust(dist(pca_df, method = 'euclidean') , method = 'ward.D')
y_hc = cutree(hc , 3)


library(cluster)
clusplot(pca_df,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = FALSE,
         main = paste('CLUSTERS'),
         xlab = 'PCA1',
         ylab = 'PCA2')


#Since in Hierarchical clustering the clusters are overlapping , we will drop this model.



#Interpreting Clusters made by K-Means Classifier
colnames(my_train)
kpi_names = c("MONTHLY_AVG_PURCHASE", 
              "MONTHLY_AVG_CASH_ADVANCE",
              "PTMP","LIMIT_USAGE", "oneoff" , "installment")
clst_names = c('CL1','CL2','CL3')
cluster_collection = data.frame()

for(j in clst_names){
  for(i in kpi_names){
    if(j == 'CL1'){cluster_collection[i,j] = mean(cluster1[,i])}
    if(j == 'CL2'){cluster_collection[i,j] = mean(cluster2[,i])}
    if(j == 'CL3'){cluster_collection[i,j] = mean(cluster3[,i])}
    
  }  
}
cluster_collection$row_names = rownames(cluster_collection)



#barplot(height = cluster_collection$CL1, names = cluster_collection$row_names, col = NULL, main = NULL)

#par(mfrow=c(2,2))


ggplot(cluster_collection, aes(x=row_names, y=CL1))+
  geom_bar(aes(fill=row_names),   # fill depends on cond2
           stat="identity",
           colour="black", 
           width = 0.5, position = "dodge")+
  theme(axis.text.x = element_text(face="bold", angle=13,size = 7),legend.position = 'none')



#Preparing a seperate dataframe consisting of all the clusters for VISUALIZING them together
a = data.frame(cluster_collection$CL1,cluster_collection$row_names)
colnames(a) = c('values','row_names')
b = data.frame(cluster_collection$CL2,cluster_collection$row_names)
colnames(b) = c('values','row_names')
c = data.frame(cluster_collection$CL3,cluster_collection$row_names)
colnames(c) = c('values','row_names')

temp = data.frame(1,'seperate_clusters')      #this will be used to seperate the clusters in the visualization
colnames(temp) = c('values', 'row_names')
rbind(a,temp,b,temp,c)
grp_cluster = data.frame()
grp_cluster = rbind(a,temp,b,temp,c)
grp_cluster$rcode = c(1:20)


ggplot(grp_cluster, aes(x=rcode, y=values))+
  geom_bar(aes(fill=row_names),   # fill depends on bar type
           stat="identity",
           colour="black", 
           width = 0.5, position = "dodge")+
  theme(axis.text.x = element_text(face="bold", angle=13,size = 7))
