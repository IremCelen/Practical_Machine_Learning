# Practical Machine Learning Course Project
Irem Celen  
August 23, 2017  

### <span style="color:blue">Overview</span>
#### <span style="color:green">Background </span>

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

#### <span style="color:green">Data</span>

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 



### <span style="color:blue">Read Training and Testing Data</span>

Loading the necessary libraries

```r
library(RCurl)
library(caret)
library(doParallel)
```

Loading the Data

```r
get_training <- getURL('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', ssl.verifyhost=FALSE, ssl.verifypeer=FALSE)
get_testing <- getURL('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', 
                  ssl.verifyhost=FALSE, ssl.verifypeer=FALSE)
training <- read.csv(textConnection(get_training), header=TRUE, na.strings = c("", "NA"))
testing <- read.csv(textConnection(get_testing), header=TRUE, na.strings = c("", "NA"))
```

### <span style="color:blue">Data Cleaning</span>

First, the data is examined to see the missing values:


```r
unique(colSums(is.na(training)))
```

```
## [1]     0 19216
```

Apperantly, the majority of the data is missing for all the columns with missing values. Therefore, those columns are excluded from the analysis.


```r
dim(training)
```

```
## [1] 19622   160
```

```r
training_cleaned <- training[,colSums(is.na(training)) == 0] 
dim(training_cleaned)
```

```
## [1] 19622    60
```

```r
dim(testing)
```

```
## [1]  20 160
```

```r
testing_cleaned <- testing[,colSums(is.na(testing)) == 0] 
dim(testing_cleaned)
```

```
## [1] 20 60
```
The remaining data sets consist of 60 variables which is still a high number. 
Zero covariates will be removed from the analysis, and user_name and cvtd_timestamp variables will be discarded.


```r
nzv <- nearZeroVar(training_cleaned, saveMetrics = TRUE)
training_cleaned2 <- training_cleaned[,-nzv$nzv]
training_final <- training_cleaned2[,-c(1,4)]

#Remove the same variables in the testing data
nzvt <- nearZeroVar(testing_cleaned, saveMetrics = TRUE)
testing_cleaned2 <- testing_cleaned[,-nzvt$nzv]
testing_final <- testing_cleaned2[,-c(1,4,59)] #problem_id doesn't exist in the training data

#A new set of training and validation sets should be created from the existing training data.
#75% of the data will be assigned to the new training set and the rest will be the testing set.

set.seed(19) #to assure reproducability
inTrain <- createDataPartition(training_final$classe, p = 0.75, list = FALSE)
train_new <- training_final[inTrain,]
validation <- training_final[-inTrain,] 
```

### <span style="color:blue"> Data Modeling </span>

For this exercise, three different models will be attempted on the data. If needed, predictors will be combined to see if a higher accuracy can be achieved.

#### <span style="color:green"> 1. Random Forest </span>
Because of its high accuracy, first, random forest will be performed with four-fold cross-validation.

```r
cl <- makeCluster(4)
registerDoParallel(cl)
model_rf <- train(classe~., method="rf", data=train_new, trControl=trainControl(method = "cv", number=4))
stopCluster(cl)
registerDoSEQ()
```
#### <span style="color:green"> 2. Linear Discriminant Analysis </span>
Since the discrimination of different groups can be achieved with high accuracy by using linear discriminant analysis, this model will also be tested with four-fold cross-validation.

```r
model_lda <- train(classe~., method="lda", data=train_new, trControl=trainControl(method = "cv", number=4))
```
#### <span style="color:green"> 3. Boosted Trees </span>
As boosted trees can give a better accuracy than random trees by adding new trees that compliments the already built ones, it will also be used with four-fold cross-validation.

```r
cl <- makeCluster(4)
registerDoParallel(cl) #to parallelize the process
model_gbm <- train(classe~., method="gbm", data=train_new)
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2459
##      2        1.4544             nan     0.1000    0.1718
##      3        1.3475             nan     0.1000    0.1371
##      4        1.2639             nan     0.1000    0.1097
##      5        1.1930             nan     0.1000    0.0941
##      6        1.1325             nan     0.1000    0.0860
##      7        1.0786             nan     0.1000    0.0755
##      8        1.0317             nan     0.1000    0.0901
##      9        0.9778             nan     0.1000    0.0803
##     10        0.9293             nan     0.1000    0.0698
##     20        0.6190             nan     0.1000    0.0354
##     40        0.3332             nan     0.1000    0.0080
##     60        0.2044             nan     0.1000    0.0106
##     80        0.1311             nan     0.1000    0.0040
##    100        0.0911             nan     0.1000    0.0028
##    120        0.0655             nan     0.1000    0.0012
##    140        0.0488             nan     0.1000    0.0007
##    150        0.0426             nan     0.1000    0.0005
```

```r
stopCluster(cl)
registerDoSEQ()
```

### <span style="color:blue"> Confusion Matrix </span>

```r
pred_rf <- predict(model_rf, validation)
pred_lda <- predict(model_lda, validation)
pred_gbm <- predict(model_gbm, validation)

#Accuracy for Random Forest:
confusionMatrix(validation$classe, pred_rf)$overall['Accuracy']
```

```
##  Accuracy 
## 0.9989804
```

```r
#Accuracy for Linear Discriminant Analysis (LDA):
confusionMatrix(validation$classe, pred_lda)$overall['Accuracy']
```

```
##  Accuracy 
## 0.7120718
```

```r
#Accuracy for Boosted Trees (GBM):
confusionMatrix(validation$classe, pred_gbm)$overall['Accuracy']
```

```
## Accuracy 
## 0.995106
```

#### <span style="color:green"> Out of Sample Error Estimate </span>
The estimated out of sample error (1-Accuracy) for the models are as following: 
LDA: 28.8%
GBM: 0.5%
Random Forest: 0.1%

Based on the results, random forest and boosted trees gave a high accuracy and low out of sample error estimates. The random forest model showed the best accuracy rate of 99.9% and so the lowest out of sample error. Since random forest has already given a very high accuracy, the models will not be combined for achieving a better accuracy, and random forest method will be used as the final model.

### <span style="color:blue"> Prediction Quiz Results </span>


```r
predict_final <- predict(model_rf, testing_final)
predict_final
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
