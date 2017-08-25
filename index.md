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


```r
model_rf <- train(classe~., method="rf", data=train_new)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```
#### <span style="color:green"> 2. Linear Discriminant Analysis </span>

```r
model_lda <- train(classe~., method="lda", data=train_new)
```

```
## Loading required package: MASS
```
#### <span style="color:green"> 3. Boosted Trees </span>

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1258
##      2        1.5248             nan     0.1000    0.0857
##      3        1.4680             nan     0.1000    0.0655
##      4        1.4243             nan     0.1000    0.0539
##      5        1.3895             nan     0.1000    0.0476
##      6        1.3580             nan     0.1000    0.0459
##      7        1.3293             nan     0.1000    0.0402
##      8        1.3036             nan     0.1000    0.0399
##      9        1.2774             nan     0.1000    0.0348
##     10        1.2557             nan     0.1000    0.0319
##     20        1.0850             nan     0.1000    0.0222
##     40        0.8857             nan     0.1000    0.0116
##     60        0.7529             nan     0.1000    0.0080
##     80        0.6498             nan     0.1000    0.0076
##    100        0.5678             nan     0.1000    0.0052
##    120        0.5029             nan     0.1000    0.0044
##    140        0.4494             nan     0.1000    0.0030
##    150        0.4256             nan     0.1000    0.0039
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1867
##      2        1.4883             nan     0.1000    0.1404
##      3        1.3991             nan     0.1000    0.1021
##      4        1.3339             nan     0.1000    0.0878
##      5        1.2787             nan     0.1000    0.0800
##      6        1.2282             nan     0.1000    0.0681
##      7        1.1851             nan     0.1000    0.0703
##      8        1.1420             nan     0.1000    0.0617
##      9        1.1036             nan     0.1000    0.0544
##     10        1.0703             nan     0.1000    0.0415
##     20        0.8111             nan     0.1000    0.0324
##     40        0.5243             nan     0.1000    0.0149
##     60        0.3640             nan     0.1000    0.0058
##     80        0.2681             nan     0.1000    0.0068
##    100        0.1970             nan     0.1000    0.0027
##    120        0.1510             nan     0.1000    0.0027
##    140        0.1182             nan     0.1000    0.0027
##    150        0.1022             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2384
##      2        1.4591             nan     0.1000    0.1675
##      3        1.3536             nan     0.1000    0.1328
##      4        1.2697             nan     0.1000    0.1187
##      5        1.1959             nan     0.1000    0.0933
##      6        1.1354             nan     0.1000    0.1060
##      7        1.0721             nan     0.1000    0.0704
##      8        1.0290             nan     0.1000    0.0776
##      9        0.9821             nan     0.1000    0.0675
##     10        0.9413             nan     0.1000    0.0695
##     20        0.6346             nan     0.1000    0.0309
##     40        0.3449             nan     0.1000    0.0114
##     60        0.2092             nan     0.1000    0.0059
##     80        0.1357             nan     0.1000    0.0038
##    100        0.0921             nan     0.1000    0.0035
##    120        0.0641             nan     0.1000    0.0013
##    140        0.0478             nan     0.1000    0.0006
##    150        0.0410             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1289
##      2        1.5225             nan     0.1000    0.0867
##      3        1.4660             nan     0.1000    0.0698
##      4        1.4208             nan     0.1000    0.0554
##      5        1.3838             nan     0.1000    0.0469
##      6        1.3539             nan     0.1000    0.0455
##      7        1.3247             nan     0.1000    0.0419
##      8        1.2984             nan     0.1000    0.0391
##      9        1.2714             nan     0.1000    0.0384
##     10        1.2483             nan     0.1000    0.0368
##     20        1.0768             nan     0.1000    0.0227
##     40        0.8716             nan     0.1000    0.0118
##     60        0.7364             nan     0.1000    0.0094
##     80        0.6372             nan     0.1000    0.0059
##    100        0.5558             nan     0.1000    0.0048
##    120        0.4945             nan     0.1000    0.0045
##    140        0.4412             nan     0.1000    0.0037
##    150        0.4183             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2014
##      2        1.4797             nan     0.1000    0.1424
##      3        1.3891             nan     0.1000    0.1075
##      4        1.3213             nan     0.1000    0.0910
##      5        1.2633             nan     0.1000    0.0833
##      6        1.2118             nan     0.1000    0.0755
##      7        1.1649             nan     0.1000    0.0590
##      8        1.1273             nan     0.1000    0.0583
##      9        1.0909             nan     0.1000    0.0460
##     10        1.0618             nan     0.1000    0.0423
##     20        0.8019             nan     0.1000    0.0319
##     40        0.5167             nan     0.1000    0.0132
##     60        0.3620             nan     0.1000    0.0110
##     80        0.2670             nan     0.1000    0.0073
##    100        0.2022             nan     0.1000    0.0033
##    120        0.1581             nan     0.1000    0.0015
##    140        0.1221             nan     0.1000    0.0020
##    150        0.1086             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2474
##      2        1.4530             nan     0.1000    0.1723
##      3        1.3441             nan     0.1000    0.1358
##      4        1.2571             nan     0.1000    0.1214
##      5        1.1816             nan     0.1000    0.0963
##      6        1.1215             nan     0.1000    0.0787
##      7        1.0714             nan     0.1000    0.0820
##      8        1.0209             nan     0.1000    0.0725
##      9        0.9765             nan     0.1000    0.0672
##     10        0.9357             nan     0.1000    0.0720
##     20        0.6241             nan     0.1000    0.0388
##     40        0.3353             nan     0.1000    0.0125
##     60        0.1994             nan     0.1000    0.0067
##     80        0.1269             nan     0.1000    0.0058
##    100        0.0883             nan     0.1000    0.0013
##    120        0.0630             nan     0.1000    0.0018
##    140        0.0475             nan     0.1000    0.0012
##    150        0.0415             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1299
##      2        1.5225             nan     0.1000    0.0870
##      3        1.4642             nan     0.1000    0.0678
##      4        1.4189             nan     0.1000    0.0520
##      5        1.3842             nan     0.1000    0.0503
##      6        1.3519             nan     0.1000    0.0448
##      7        1.3227             nan     0.1000    0.0388
##      8        1.2980             nan     0.1000    0.0450
##      9        1.2677             nan     0.1000    0.0320
##     10        1.2469             nan     0.1000    0.0379
##     20        1.0722             nan     0.1000    0.0206
##     40        0.8740             nan     0.1000    0.0113
##     60        0.7399             nan     0.1000    0.0105
##     80        0.6377             nan     0.1000    0.0057
##    100        0.5597             nan     0.1000    0.0046
##    120        0.4946             nan     0.1000    0.0048
##    140        0.4440             nan     0.1000    0.0038
##    150        0.4207             nan     0.1000    0.0036
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1930
##      2        1.4822             nan     0.1000    0.1399
##      3        1.3942             nan     0.1000    0.1113
##      4        1.3235             nan     0.1000    0.0873
##      5        1.2662             nan     0.1000    0.0824
##      6        1.2156             nan     0.1000    0.0701
##      7        1.1707             nan     0.1000    0.0649
##      8        1.1302             nan     0.1000    0.0542
##      9        1.0951             nan     0.1000    0.0554
##     10        1.0608             nan     0.1000    0.0491
##     20        0.7985             nan     0.1000    0.0312
##     40        0.5287             nan     0.1000    0.0212
##     60        0.3636             nan     0.1000    0.0091
##     80        0.2668             nan     0.1000    0.0062
##    100        0.2020             nan     0.1000    0.0036
##    120        0.1542             nan     0.1000    0.0033
##    140        0.1199             nan     0.1000    0.0017
##    150        0.1052             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2465
##      2        1.4514             nan     0.1000    0.1750
##      3        1.3399             nan     0.1000    0.1401
##      4        1.2532             nan     0.1000    0.1187
##      5        1.1780             nan     0.1000    0.0914
##      6        1.1192             nan     0.1000    0.1011
##      7        1.0582             nan     0.1000    0.0769
##      8        1.0114             nan     0.1000    0.0624
##      9        0.9714             nan     0.1000    0.0804
##     10        0.9231             nan     0.1000    0.0583
##     20        0.6148             nan     0.1000    0.0317
##     40        0.3299             nan     0.1000    0.0141
##     60        0.2060             nan     0.1000    0.0078
##     80        0.1329             nan     0.1000    0.0040
##    100        0.0904             nan     0.1000    0.0027
##    120        0.0653             nan     0.1000    0.0017
##    140        0.0477             nan     0.1000    0.0012
##    150        0.0409             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1277
##      2        1.5228             nan     0.1000    0.0885
##      3        1.4643             nan     0.1000    0.0682
##      4        1.4187             nan     0.1000    0.0527
##      5        1.3838             nan     0.1000    0.0532
##      6        1.3499             nan     0.1000    0.0456
##      7        1.3203             nan     0.1000    0.0395
##      8        1.2943             nan     0.1000    0.0349
##      9        1.2717             nan     0.1000    0.0347
##     10        1.2478             nan     0.1000    0.0325
##     20        1.0783             nan     0.1000    0.0181
##     40        0.8787             nan     0.1000    0.0096
##     60        0.7428             nan     0.1000    0.0092
##     80        0.6429             nan     0.1000    0.0087
##    100        0.5630             nan     0.1000    0.0063
##    120        0.4978             nan     0.1000    0.0037
##    140        0.4456             nan     0.1000    0.0044
##    150        0.4230             nan     0.1000    0.0036
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1956
##      2        1.4841             nan     0.1000    0.1354
##      3        1.3967             nan     0.1000    0.1115
##      4        1.3273             nan     0.1000    0.0871
##      5        1.2706             nan     0.1000    0.0746
##      6        1.2222             nan     0.1000    0.0720
##      7        1.1764             nan     0.1000    0.0671
##      8        1.1344             nan     0.1000    0.0558
##      9        1.0992             nan     0.1000    0.0543
##     10        1.0657             nan     0.1000    0.0434
##     20        0.8031             nan     0.1000    0.0300
##     40        0.5283             nan     0.1000    0.0234
##     60        0.3677             nan     0.1000    0.0085
##     80        0.2687             nan     0.1000    0.0063
##    100        0.2044             nan     0.1000    0.0048
##    120        0.1558             nan     0.1000    0.0048
##    140        0.1220             nan     0.1000    0.0030
##    150        0.1063             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2444
##      2        1.4535             nan     0.1000    0.1665
##      3        1.3476             nan     0.1000    0.1477
##      4        1.2562             nan     0.1000    0.1099
##      5        1.1873             nan     0.1000    0.0948
##      6        1.1278             nan     0.1000    0.0835
##      7        1.0731             nan     0.1000    0.0768
##      8        1.0254             nan     0.1000    0.0882
##      9        0.9731             nan     0.1000    0.0680
##     10        0.9313             nan     0.1000    0.0645
##     20        0.6261             nan     0.1000    0.0454
##     40        0.3306             nan     0.1000    0.0134
##     60        0.2055             nan     0.1000    0.0066
##     80        0.1350             nan     0.1000    0.0043
##    100        0.0906             nan     0.1000    0.0020
##    120        0.0666             nan     0.1000    0.0020
##    140        0.0500             nan     0.1000    0.0010
##    150        0.0442             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1258
##      2        1.5260             nan     0.1000    0.0858
##      3        1.4692             nan     0.1000    0.0648
##      4        1.4257             nan     0.1000    0.0529
##      5        1.3915             nan     0.1000    0.0482
##      6        1.3603             nan     0.1000    0.0467
##      7        1.3309             nan     0.1000    0.0385
##      8        1.3062             nan     0.1000    0.0396
##      9        1.2803             nan     0.1000    0.0334
##     10        1.2596             nan     0.1000    0.0357
##     20        1.0889             nan     0.1000    0.0203
##     40        0.8875             nan     0.1000    0.0105
##     60        0.7521             nan     0.1000    0.0076
##     80        0.6481             nan     0.1000    0.0068
##    100        0.5703             nan     0.1000    0.0049
##    120        0.5050             nan     0.1000    0.0042
##    140        0.4519             nan     0.1000    0.0035
##    150        0.4295             nan     0.1000    0.0036
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1864
##      2        1.4886             nan     0.1000    0.1307
##      3        1.4045             nan     0.1000    0.1065
##      4        1.3369             nan     0.1000    0.0875
##      5        1.2813             nan     0.1000    0.0749
##      6        1.2331             nan     0.1000    0.0805
##      7        1.1848             nan     0.1000    0.0715
##      8        1.1412             nan     0.1000    0.0594
##      9        1.1044             nan     0.1000    0.0561
##     10        1.0705             nan     0.1000    0.0566
##     20        0.8039             nan     0.1000    0.0284
##     40        0.5318             nan     0.1000    0.0309
##     60        0.3668             nan     0.1000    0.0096
##     80        0.2609             nan     0.1000    0.0043
##    100        0.1998             nan     0.1000    0.0054
##    120        0.1492             nan     0.1000    0.0034
##    140        0.1162             nan     0.1000    0.0011
##    150        0.1037             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2400
##      2        1.4601             nan     0.1000    0.1682
##      3        1.3543             nan     0.1000    0.1314
##      4        1.2729             nan     0.1000    0.1126
##      5        1.2019             nan     0.1000    0.0944
##      6        1.1423             nan     0.1000    0.0949
##      7        1.0844             nan     0.1000    0.0713
##      8        1.0403             nan     0.1000    0.0642
##      9        1.0008             nan     0.1000    0.0709
##     10        0.9578             nan     0.1000    0.0695
##     20        0.6323             nan     0.1000    0.0286
##     40        0.3407             nan     0.1000    0.0145
##     60        0.2108             nan     0.1000    0.0042
##     80        0.1380             nan     0.1000    0.0033
##    100        0.0964             nan     0.1000    0.0027
##    120        0.0701             nan     0.1000    0.0012
##    140        0.0513             nan     0.1000    0.0010
##    150        0.0449             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1335
##      2        1.5211             nan     0.1000    0.0920
##      3        1.4616             nan     0.1000    0.0684
##      4        1.4168             nan     0.1000    0.0560
##      5        1.3804             nan     0.1000    0.0487
##      6        1.3491             nan     0.1000    0.0473
##      7        1.3195             nan     0.1000    0.0382
##      8        1.2951             nan     0.1000    0.0419
##      9        1.2674             nan     0.1000    0.0364
##     10        1.2443             nan     0.1000    0.0299
##     20        1.0743             nan     0.1000    0.0193
##     40        0.8753             nan     0.1000    0.0142
##     60        0.7434             nan     0.1000    0.0073
##     80        0.6425             nan     0.1000    0.0064
##    100        0.5614             nan     0.1000    0.0037
##    120        0.4975             nan     0.1000    0.0046
##    140        0.4477             nan     0.1000    0.0032
##    150        0.4254             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1842
##      2        1.4863             nan     0.1000    0.1369
##      3        1.3981             nan     0.1000    0.1071
##      4        1.3295             nan     0.1000    0.0910
##      5        1.2703             nan     0.1000    0.0829
##      6        1.2180             nan     0.1000    0.0699
##      7        1.1736             nan     0.1000    0.0617
##      8        1.1337             nan     0.1000    0.0597
##      9        1.0971             nan     0.1000    0.0538
##     10        1.0630             nan     0.1000    0.0494
##     20        0.7968             nan     0.1000    0.0271
##     40        0.5171             nan     0.1000    0.0203
##     60        0.3598             nan     0.1000    0.0135
##     80        0.2601             nan     0.1000    0.0088
##    100        0.1969             nan     0.1000    0.0045
##    120        0.1473             nan     0.1000    0.0023
##    140        0.1180             nan     0.1000    0.0025
##    150        0.1048             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2417
##      2        1.4529             nan     0.1000    0.1715
##      3        1.3438             nan     0.1000    0.1460
##      4        1.2544             nan     0.1000    0.1107
##      5        1.1842             nan     0.1000    0.0929
##      6        1.1231             nan     0.1000    0.0766
##      7        1.0740             nan     0.1000    0.0921
##      8        1.0171             nan     0.1000    0.0709
##      9        0.9732             nan     0.1000    0.0623
##     10        0.9334             nan     0.1000    0.0668
##     20        0.6281             nan     0.1000    0.0414
##     40        0.3349             nan     0.1000    0.0164
##     60        0.2068             nan     0.1000    0.0083
##     80        0.1316             nan     0.1000    0.0027
##    100        0.0922             nan     0.1000    0.0019
##    120        0.0674             nan     0.1000    0.0013
##    140        0.0489             nan     0.1000    0.0005
##    150        0.0433             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1271
##      2        1.5220             nan     0.1000    0.0901
##      3        1.4620             nan     0.1000    0.0667
##      4        1.4170             nan     0.1000    0.0557
##      5        1.3806             nan     0.1000    0.0509
##      6        1.3485             nan     0.1000    0.0453
##      7        1.3193             nan     0.1000    0.0416
##      8        1.2931             nan     0.1000    0.0379
##      9        1.2699             nan     0.1000    0.0364
##     10        1.2450             nan     0.1000    0.0363
##     20        1.0723             nan     0.1000    0.0210
##     40        0.8659             nan     0.1000    0.0127
##     60        0.7336             nan     0.1000    0.0081
##     80        0.6355             nan     0.1000    0.0065
##    100        0.5555             nan     0.1000    0.0061
##    120        0.4928             nan     0.1000    0.0035
##    140        0.4433             nan     0.1000    0.0030
##    150        0.4184             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1952
##      2        1.4855             nan     0.1000    0.1342
##      3        1.3980             nan     0.1000    0.1076
##      4        1.3295             nan     0.1000    0.0910
##      5        1.2716             nan     0.1000    0.0802
##      6        1.2216             nan     0.1000    0.0804
##      7        1.1721             nan     0.1000    0.0637
##      8        1.1324             nan     0.1000    0.0527
##      9        1.0991             nan     0.1000    0.0609
##     10        1.0617             nan     0.1000    0.0439
##     20        0.7900             nan     0.1000    0.0348
##     40        0.5110             nan     0.1000    0.0090
##     60        0.3549             nan     0.1000    0.0123
##     80        0.2617             nan     0.1000    0.0057
##    100        0.1953             nan     0.1000    0.0048
##    120        0.1500             nan     0.1000    0.0036
##    140        0.1184             nan     0.1000    0.0019
##    150        0.1049             nan     0.1000    0.0026
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2428
##      2        1.4562             nan     0.1000    0.1723
##      3        1.3492             nan     0.1000    0.1374
##      4        1.2630             nan     0.1000    0.1152
##      5        1.1922             nan     0.1000    0.1001
##      6        1.1310             nan     0.1000    0.0845
##      7        1.0784             nan     0.1000    0.0785
##      8        1.0303             nan     0.1000    0.0860
##      9        0.9791             nan     0.1000    0.0733
##     10        0.9351             nan     0.1000    0.0665
##     20        0.6262             nan     0.1000    0.0502
##     40        0.3458             nan     0.1000    0.0161
##     60        0.2109             nan     0.1000    0.0077
##     80        0.1345             nan     0.1000    0.0034
##    100        0.0943             nan     0.1000    0.0025
##    120        0.0690             nan     0.1000    0.0013
##    140        0.0519             nan     0.1000    0.0007
##    150        0.0456             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1289
##      2        1.5226             nan     0.1000    0.0891
##      3        1.4625             nan     0.1000    0.0684
##      4        1.4174             nan     0.1000    0.0548
##      5        1.3806             nan     0.1000    0.0479
##      6        1.3490             nan     0.1000    0.0490
##      7        1.3186             nan     0.1000    0.0388
##      8        1.2932             nan     0.1000    0.0369
##      9        1.2698             nan     0.1000    0.0347
##     10        1.2477             nan     0.1000    0.0359
##     20        1.0704             nan     0.1000    0.0205
##     40        0.8714             nan     0.1000    0.0133
##     60        0.7395             nan     0.1000    0.0097
##     80        0.6410             nan     0.1000    0.0073
##    100        0.5601             nan     0.1000    0.0039
##    120        0.4962             nan     0.1000    0.0038
##    140        0.4445             nan     0.1000    0.0046
##    150        0.4209             nan     0.1000    0.0032
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1911
##      2        1.4832             nan     0.1000    0.1322
##      3        1.3954             nan     0.1000    0.1070
##      4        1.3263             nan     0.1000    0.0937
##      5        1.2666             nan     0.1000    0.0703
##      6        1.2212             nan     0.1000    0.0772
##      7        1.1728             nan     0.1000    0.0761
##      8        1.1263             nan     0.1000    0.0647
##      9        1.0869             nan     0.1000    0.0545
##     10        1.0536             nan     0.1000    0.0501
##     20        0.7939             nan     0.1000    0.0232
##     40        0.5130             nan     0.1000    0.0164
##     60        0.3619             nan     0.1000    0.0083
##     80        0.2594             nan     0.1000    0.0048
##    100        0.1993             nan     0.1000    0.0048
##    120        0.1474             nan     0.1000    0.0019
##    140        0.1163             nan     0.1000    0.0028
##    150        0.1033             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2421
##      2        1.4526             nan     0.1000    0.1657
##      3        1.3486             nan     0.1000    0.1328
##      4        1.2651             nan     0.1000    0.1172
##      5        1.1926             nan     0.1000    0.0971
##      6        1.1319             nan     0.1000    0.0945
##      7        1.0732             nan     0.1000    0.0846
##      8        1.0203             nan     0.1000    0.0639
##      9        0.9795             nan     0.1000    0.0766
##     10        0.9326             nan     0.1000    0.0614
##     20        0.6251             nan     0.1000    0.0330
##     40        0.3335             nan     0.1000    0.0114
##     60        0.1997             nan     0.1000    0.0106
##     80        0.1282             nan     0.1000    0.0042
##    100        0.0875             nan     0.1000    0.0032
##    120        0.0621             nan     0.1000    0.0019
##    140        0.0459             nan     0.1000    0.0006
##    150        0.0404             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1332
##      2        1.5216             nan     0.1000    0.0911
##      3        1.4619             nan     0.1000    0.0672
##      4        1.4169             nan     0.1000    0.0554
##      5        1.3799             nan     0.1000    0.0469
##      6        1.3486             nan     0.1000    0.0493
##      7        1.3180             nan     0.1000    0.0401
##      8        1.2924             nan     0.1000    0.0413
##      9        1.2649             nan     0.1000    0.0330
##     10        1.2434             nan     0.1000    0.0345
##     20        1.0702             nan     0.1000    0.0187
##     40        0.8724             nan     0.1000    0.0141
##     60        0.7384             nan     0.1000    0.0102
##     80        0.6356             nan     0.1000    0.0059
##    100        0.5586             nan     0.1000    0.0061
##    120        0.4973             nan     0.1000    0.0050
##    140        0.4442             nan     0.1000    0.0039
##    150        0.4199             nan     0.1000    0.0035
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1971
##      2        1.4826             nan     0.1000    0.1377
##      3        1.3946             nan     0.1000    0.1052
##      4        1.3257             nan     0.1000    0.0912
##      5        1.2675             nan     0.1000    0.0764
##      6        1.2190             nan     0.1000    0.0706
##      7        1.1743             nan     0.1000    0.0718
##      8        1.1304             nan     0.1000    0.0588
##      9        1.0944             nan     0.1000    0.0577
##     10        1.0577             nan     0.1000    0.0424
##     20        0.8017             nan     0.1000    0.0364
##     40        0.5314             nan     0.1000    0.0256
##     60        0.3648             nan     0.1000    0.0076
##     80        0.2641             nan     0.1000    0.0083
##    100        0.1938             nan     0.1000    0.0045
##    120        0.1481             nan     0.1000    0.0024
##    140        0.1173             nan     0.1000    0.0029
##    150        0.1052             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2414
##      2        1.4538             nan     0.1000    0.1739
##      3        1.3464             nan     0.1000    0.1337
##      4        1.2620             nan     0.1000    0.1095
##      5        1.1923             nan     0.1000    0.0947
##      6        1.1330             nan     0.1000    0.0873
##      7        1.0790             nan     0.1000    0.0916
##      8        1.0229             nan     0.1000    0.0790
##      9        0.9738             nan     0.1000    0.0800
##     10        0.9255             nan     0.1000    0.0672
##     20        0.6137             nan     0.1000    0.0380
##     40        0.3353             nan     0.1000    0.0154
##     60        0.2020             nan     0.1000    0.0047
##     80        0.1310             nan     0.1000    0.0035
##    100        0.0862             nan     0.1000    0.0039
##    120        0.0614             nan     0.1000    0.0011
##    140        0.0456             nan     0.1000    0.0006
##    150        0.0401             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1334
##      2        1.5216             nan     0.1000    0.0827
##      3        1.4657             nan     0.1000    0.0686
##      4        1.4196             nan     0.1000    0.0555
##      5        1.3838             nan     0.1000    0.0476
##      6        1.3523             nan     0.1000    0.0480
##      7        1.3227             nan     0.1000    0.0427
##      8        1.2958             nan     0.1000    0.0334
##      9        1.2741             nan     0.1000    0.0391
##     10        1.2483             nan     0.1000    0.0332
##     20        1.0801             nan     0.1000    0.0181
##     40        0.8803             nan     0.1000    0.0111
##     60        0.7466             nan     0.1000    0.0117
##     80        0.6432             nan     0.1000    0.0079
##    100        0.5637             nan     0.1000    0.0042
##    120        0.5001             nan     0.1000    0.0038
##    140        0.4478             nan     0.1000    0.0033
##    150        0.4253             nan     0.1000    0.0029
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1878
##      2        1.4877             nan     0.1000    0.1331
##      3        1.4012             nan     0.1000    0.1016
##      4        1.3349             nan     0.1000    0.0868
##      5        1.2788             nan     0.1000    0.0854
##      6        1.2244             nan     0.1000    0.0781
##      7        1.1751             nan     0.1000    0.0651
##      8        1.1338             nan     0.1000    0.0541
##      9        1.0991             nan     0.1000    0.0508
##     10        1.0670             nan     0.1000    0.0563
##     20        0.8077             nan     0.1000    0.0329
##     40        0.5217             nan     0.1000    0.0171
##     60        0.3661             nan     0.1000    0.0096
##     80        0.2600             nan     0.1000    0.0045
##    100        0.1989             nan     0.1000    0.0032
##    120        0.1511             nan     0.1000    0.0047
##    140        0.1177             nan     0.1000    0.0022
##    150        0.1044             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2337
##      2        1.4585             nan     0.1000    0.1679
##      3        1.3533             nan     0.1000    0.1331
##      4        1.2694             nan     0.1000    0.1151
##      5        1.1964             nan     0.1000    0.1063
##      6        1.1310             nan     0.1000    0.0857
##      7        1.0771             nan     0.1000    0.0709
##      8        1.0319             nan     0.1000    0.0793
##      9        0.9834             nan     0.1000    0.0651
##     10        0.9430             nan     0.1000    0.0738
##     20        0.6328             nan     0.1000    0.0441
##     40        0.3459             nan     0.1000    0.0124
##     60        0.2108             nan     0.1000    0.0084
##     80        0.1337             nan     0.1000    0.0053
##    100        0.0938             nan     0.1000    0.0021
##    120        0.0669             nan     0.1000    0.0015
##    140        0.0505             nan     0.1000    0.0014
##    150        0.0436             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1291
##      2        1.5232             nan     0.1000    0.0876
##      3        1.4644             nan     0.1000    0.0673
##      4        1.4199             nan     0.1000    0.0551
##      5        1.3840             nan     0.1000    0.0494
##      6        1.3527             nan     0.1000    0.0452
##      7        1.3242             nan     0.1000    0.0331
##      8        1.3009             nan     0.1000    0.0421
##      9        1.2729             nan     0.1000    0.0368
##     10        1.2491             nan     0.1000    0.0357
##     20        1.0726             nan     0.1000    0.0206
##     40        0.8706             nan     0.1000    0.0115
##     60        0.7364             nan     0.1000    0.0070
##     80        0.6354             nan     0.1000    0.0079
##    100        0.5591             nan     0.1000    0.0044
##    120        0.4941             nan     0.1000    0.0037
##    140        0.4418             nan     0.1000    0.0033
##    150        0.4185             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1871
##      2        1.4869             nan     0.1000    0.1351
##      3        1.3991             nan     0.1000    0.1158
##      4        1.3255             nan     0.1000    0.0874
##      5        1.2697             nan     0.1000    0.0856
##      6        1.2153             nan     0.1000    0.0700
##      7        1.1700             nan     0.1000    0.0715
##      8        1.1255             nan     0.1000    0.0553
##      9        1.0912             nan     0.1000    0.0507
##     10        1.0594             nan     0.1000    0.0484
##     20        0.7955             nan     0.1000    0.0315
##     40        0.5117             nan     0.1000    0.0150
##     60        0.3562             nan     0.1000    0.0129
##     80        0.2642             nan     0.1000    0.0045
##    100        0.1995             nan     0.1000    0.0046
##    120        0.1531             nan     0.1000    0.0032
##    140        0.1168             nan     0.1000    0.0030
##    150        0.1033             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2457
##      2        1.4537             nan     0.1000    0.1824
##      3        1.3411             nan     0.1000    0.1309
##      4        1.2593             nan     0.1000    0.1075
##      5        1.1897             nan     0.1000    0.0989
##      6        1.1280             nan     0.1000    0.0935
##      7        1.0701             nan     0.1000    0.0802
##      8        1.0212             nan     0.1000    0.0837
##      9        0.9703             nan     0.1000    0.0690
##     10        0.9277             nan     0.1000    0.0550
##     20        0.6211             nan     0.1000    0.0344
##     40        0.3414             nan     0.1000    0.0105
##     60        0.2084             nan     0.1000    0.0107
##     80        0.1380             nan     0.1000    0.0044
##    100        0.0921             nan     0.1000    0.0026
##    120        0.0666             nan     0.1000    0.0016
##    140        0.0495             nan     0.1000    0.0014
##    150        0.0434             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1274
##      2        1.5239             nan     0.1000    0.0838
##      3        1.4666             nan     0.1000    0.0663
##      4        1.4226             nan     0.1000    0.0585
##      5        1.3854             nan     0.1000    0.0498
##      6        1.3530             nan     0.1000    0.0472
##      7        1.3230             nan     0.1000    0.0421
##      8        1.2966             nan     0.1000    0.0375
##      9        1.2728             nan     0.1000    0.0395
##     10        1.2466             nan     0.1000    0.0339
##     20        1.0756             nan     0.1000    0.0229
##     40        0.8715             nan     0.1000    0.0102
##     60        0.7413             nan     0.1000    0.0084
##     80        0.6395             nan     0.1000    0.0063
##    100        0.5616             nan     0.1000    0.0067
##    120        0.4989             nan     0.1000    0.0055
##    140        0.4477             nan     0.1000    0.0038
##    150        0.4232             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1856
##      2        1.4870             nan     0.1000    0.1404
##      3        1.3976             nan     0.1000    0.1039
##      4        1.3308             nan     0.1000    0.0916
##      5        1.2730             nan     0.1000    0.0803
##      6        1.2222             nan     0.1000    0.0678
##      7        1.1799             nan     0.1000    0.0700
##      8        1.1369             nan     0.1000    0.0595
##      9        1.1007             nan     0.1000    0.0559
##     10        1.0661             nan     0.1000    0.0529
##     20        0.7882             nan     0.1000    0.0289
##     40        0.5106             nan     0.1000    0.0180
##     60        0.3618             nan     0.1000    0.0078
##     80        0.2647             nan     0.1000    0.0075
##    100        0.1962             nan     0.1000    0.0049
##    120        0.1493             nan     0.1000    0.0036
##    140        0.1156             nan     0.1000    0.0014
##    150        0.1025             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2390
##      2        1.4539             nan     0.1000    0.1779
##      3        1.3426             nan     0.1000    0.1333
##      4        1.2584             nan     0.1000    0.1090
##      5        1.1898             nan     0.1000    0.0969
##      6        1.1293             nan     0.1000    0.0856
##      7        1.0745             nan     0.1000    0.0904
##      8        1.0199             nan     0.1000    0.0667
##      9        0.9779             nan     0.1000    0.0648
##     10        0.9379             nan     0.1000    0.0647
##     20        0.6209             nan     0.1000    0.0318
##     40        0.3381             nan     0.1000    0.0159
##     60        0.2094             nan     0.1000    0.0077
##     80        0.1349             nan     0.1000    0.0058
##    100        0.0934             nan     0.1000    0.0015
##    120        0.0668             nan     0.1000    0.0013
##    140        0.0498             nan     0.1000    0.0009
##    150        0.0441             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1304
##      2        1.5227             nan     0.1000    0.0866
##      3        1.4648             nan     0.1000    0.0669
##      4        1.4195             nan     0.1000    0.0530
##      5        1.3844             nan     0.1000    0.0466
##      6        1.3538             nan     0.1000    0.0450
##      7        1.3242             nan     0.1000    0.0387
##      8        1.2990             nan     0.1000    0.0383
##      9        1.2723             nan     0.1000    0.0394
##     10        1.2488             nan     0.1000    0.0381
##     20        1.0735             nan     0.1000    0.0210
##     40        0.8710             nan     0.1000    0.0103
##     60        0.7372             nan     0.1000    0.0096
##     80        0.6316             nan     0.1000    0.0053
##    100        0.5544             nan     0.1000    0.0056
##    120        0.4909             nan     0.1000    0.0054
##    140        0.4402             nan     0.1000    0.0031
##    150        0.4182             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1922
##      2        1.4834             nan     0.1000    0.1335
##      3        1.3959             nan     0.1000    0.1103
##      4        1.3263             nan     0.1000    0.0848
##      5        1.2718             nan     0.1000    0.0868
##      6        1.2181             nan     0.1000    0.0744
##      7        1.1718             nan     0.1000    0.0672
##      8        1.1299             nan     0.1000    0.0685
##      9        1.0882             nan     0.1000    0.0643
##     10        1.0507             nan     0.1000    0.0513
##     20        0.7978             nan     0.1000    0.0246
##     40        0.5221             nan     0.1000    0.0164
##     60        0.3577             nan     0.1000    0.0075
##     80        0.2575             nan     0.1000    0.0047
##    100        0.1927             nan     0.1000    0.0045
##    120        0.1438             nan     0.1000    0.0017
##    140        0.1103             nan     0.1000    0.0030
##    150        0.0986             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2456
##      2        1.4527             nan     0.1000    0.1741
##      3        1.3434             nan     0.1000    0.1367
##      4        1.2578             nan     0.1000    0.1149
##      5        1.1853             nan     0.1000    0.0904
##      6        1.1278             nan     0.1000    0.0872
##      7        1.0737             nan     0.1000    0.0819
##      8        1.0235             nan     0.1000    0.0737
##      9        0.9774             nan     0.1000    0.0657
##     10        0.9367             nan     0.1000    0.0662
##     20        0.6244             nan     0.1000    0.0410
##     40        0.3370             nan     0.1000    0.0176
##     60        0.2039             nan     0.1000    0.0059
##     80        0.1307             nan     0.1000    0.0050
##    100        0.0895             nan     0.1000    0.0016
##    120        0.0636             nan     0.1000    0.0012
##    140        0.0465             nan     0.1000    0.0011
##    150        0.0407             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1253
##      2        1.5222             nan     0.1000    0.0904
##      3        1.4631             nan     0.1000    0.0675
##      4        1.4178             nan     0.1000    0.0551
##      5        1.3813             nan     0.1000    0.0510
##      6        1.3487             nan     0.1000    0.0380
##      7        1.3230             nan     0.1000    0.0408
##      8        1.2967             nan     0.1000    0.0435
##      9        1.2686             nan     0.1000    0.0312
##     10        1.2485             nan     0.1000    0.0318
##     20        1.0733             nan     0.1000    0.0183
##     40        0.8720             nan     0.1000    0.0105
##     60        0.7353             nan     0.1000    0.0095
##     80        0.6345             nan     0.1000    0.0068
##    100        0.5578             nan     0.1000    0.0058
##    120        0.4935             nan     0.1000    0.0042
##    140        0.4418             nan     0.1000    0.0035
##    150        0.4198             nan     0.1000    0.0040
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1969
##      2        1.4819             nan     0.1000    0.1347
##      3        1.3932             nan     0.1000    0.1035
##      4        1.3254             nan     0.1000    0.0941
##      5        1.2658             nan     0.1000    0.0782
##      6        1.2165             nan     0.1000    0.0699
##      7        1.1710             nan     0.1000    0.0725
##      8        1.1267             nan     0.1000    0.0586
##      9        1.0910             nan     0.1000    0.0624
##     10        1.0532             nan     0.1000    0.0446
##     20        0.7918             nan     0.1000    0.0220
##     40        0.5240             nan     0.1000    0.0159
##     60        0.3681             nan     0.1000    0.0106
##     80        0.2641             nan     0.1000    0.0065
##    100        0.1979             nan     0.1000    0.0018
##    120        0.1544             nan     0.1000    0.0035
##    140        0.1208             nan     0.1000    0.0039
##    150        0.1055             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2479
##      2        1.4515             nan     0.1000    0.1693
##      3        1.3449             nan     0.1000    0.1318
##      4        1.2613             nan     0.1000    0.1174
##      5        1.1867             nan     0.1000    0.0978
##      6        1.1264             nan     0.1000    0.0860
##      7        1.0735             nan     0.1000    0.0866
##      8        1.0207             nan     0.1000    0.0706
##      9        0.9759             nan     0.1000    0.0740
##     10        0.9311             nan     0.1000    0.0660
##     20        0.6299             nan     0.1000    0.0520
##     40        0.3367             nan     0.1000    0.0118
##     60        0.2064             nan     0.1000    0.0101
##     80        0.1356             nan     0.1000    0.0045
##    100        0.0926             nan     0.1000    0.0027
##    120        0.0655             nan     0.1000    0.0016
##    140        0.0480             nan     0.1000    0.0015
##    150        0.0412             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1290
##      2        1.5224             nan     0.1000    0.0878
##      3        1.4637             nan     0.1000    0.0702
##      4        1.4184             nan     0.1000    0.0537
##      5        1.3830             nan     0.1000    0.0538
##      6        1.3479             nan     0.1000    0.0463
##      7        1.3184             nan     0.1000    0.0424
##      8        1.2924             nan     0.1000    0.0350
##      9        1.2696             nan     0.1000    0.0408
##     10        1.2433             nan     0.1000    0.0302
##     20        1.0736             nan     0.1000    0.0181
##     40        0.8739             nan     0.1000    0.0123
##     60        0.7373             nan     0.1000    0.0096
##     80        0.6386             nan     0.1000    0.0057
##    100        0.5621             nan     0.1000    0.0055
##    120        0.4967             nan     0.1000    0.0040
##    140        0.4439             nan     0.1000    0.0033
##    150        0.4211             nan     0.1000    0.0030
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1990
##      2        1.4811             nan     0.1000    0.1343
##      3        1.3940             nan     0.1000    0.1069
##      4        1.3232             nan     0.1000    0.0895
##      5        1.2661             nan     0.1000    0.0696
##      6        1.2209             nan     0.1000    0.0786
##      7        1.1721             nan     0.1000    0.0611
##      8        1.1337             nan     0.1000    0.0659
##      9        1.0939             nan     0.1000    0.0510
##     10        1.0616             nan     0.1000    0.0472
##     20        0.8041             nan     0.1000    0.0295
##     40        0.5183             nan     0.1000    0.0121
##     60        0.3570             nan     0.1000    0.0068
##     80        0.2658             nan     0.1000    0.0068
##    100        0.2002             nan     0.1000    0.0039
##    120        0.1512             nan     0.1000    0.0013
##    140        0.1192             nan     0.1000    0.0030
##    150        0.1046             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2457
##      2        1.4529             nan     0.1000    0.1694
##      3        1.3472             nan     0.1000    0.1437
##      4        1.2573             nan     0.1000    0.1114
##      5        1.1878             nan     0.1000    0.0982
##      6        1.1272             nan     0.1000    0.0834
##      7        1.0756             nan     0.1000    0.0856
##      8        1.0231             nan     0.1000    0.0802
##      9        0.9744             nan     0.1000    0.0529
##     10        0.9405             nan     0.1000    0.0612
##     20        0.6358             nan     0.1000    0.0390
##     40        0.3385             nan     0.1000    0.0127
##     60        0.2071             nan     0.1000    0.0108
##     80        0.1345             nan     0.1000    0.0032
##    100        0.0915             nan     0.1000    0.0023
##    120        0.0658             nan     0.1000    0.0012
##    140        0.0476             nan     0.1000    0.0010
##    150        0.0415             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1312
##      2        1.5243             nan     0.1000    0.0864
##      3        1.4663             nan     0.1000    0.0682
##      4        1.4228             nan     0.1000    0.0539
##      5        1.3871             nan     0.1000    0.0516
##      6        1.3540             nan     0.1000    0.0470
##      7        1.3241             nan     0.1000    0.0370
##      8        1.2997             nan     0.1000    0.0373
##      9        1.2738             nan     0.1000    0.0343
##     10        1.2526             nan     0.1000    0.0376
##     20        1.0807             nan     0.1000    0.0208
##     40        0.8809             nan     0.1000    0.0114
##     60        0.7457             nan     0.1000    0.0107
##     80        0.6438             nan     0.1000    0.0091
##    100        0.5620             nan     0.1000    0.0051
##    120        0.4953             nan     0.1000    0.0037
##    140        0.4446             nan     0.1000    0.0029
##    150        0.4227             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1940
##      2        1.4847             nan     0.1000    0.1383
##      3        1.3974             nan     0.1000    0.1050
##      4        1.3289             nan     0.1000    0.0913
##      5        1.2712             nan     0.1000    0.0863
##      6        1.2174             nan     0.1000    0.0628
##      7        1.1769             nan     0.1000    0.0633
##      8        1.1367             nan     0.1000    0.0641
##      9        1.0971             nan     0.1000    0.0542
##     10        1.0644             nan     0.1000    0.0502
##     20        0.8035             nan     0.1000    0.0266
##     40        0.5226             nan     0.1000    0.0149
##     60        0.3584             nan     0.1000    0.0073
##     80        0.2592             nan     0.1000    0.0079
##    100        0.1950             nan     0.1000    0.0032
##    120        0.1513             nan     0.1000    0.0028
##    140        0.1176             nan     0.1000    0.0029
##    150        0.1045             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2483
##      2        1.4539             nan     0.1000    0.1689
##      3        1.3482             nan     0.1000    0.1443
##      4        1.2577             nan     0.1000    0.1065
##      5        1.1910             nan     0.1000    0.0931
##      6        1.1322             nan     0.1000    0.0932
##      7        1.0751             nan     0.1000    0.0665
##      8        1.0316             nan     0.1000    0.0918
##      9        0.9776             nan     0.1000    0.0780
##     10        0.9299             nan     0.1000    0.0627
##     20        0.6170             nan     0.1000    0.0422
##     40        0.3434             nan     0.1000    0.0142
##     60        0.2070             nan     0.1000    0.0087
##     80        0.1370             nan     0.1000    0.0055
##    100        0.0926             nan     0.1000    0.0029
##    120        0.0674             nan     0.1000    0.0025
##    140        0.0488             nan     0.1000    0.0009
##    150        0.0431             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1307
##      2        1.5242             nan     0.1000    0.0857
##      3        1.4668             nan     0.1000    0.0683
##      4        1.4231             nan     0.1000    0.0530
##      5        1.3881             nan     0.1000    0.0523
##      6        1.3546             nan     0.1000    0.0423
##      7        1.3273             nan     0.1000    0.0425
##      8        1.3010             nan     0.1000    0.0378
##      9        1.2756             nan     0.1000    0.0350
##     10        1.2532             nan     0.1000    0.0302
##     20        1.0802             nan     0.1000    0.0204
##     40        0.8781             nan     0.1000    0.0118
##     60        0.7450             nan     0.1000    0.0084
##     80        0.6429             nan     0.1000    0.0064
##    100        0.5609             nan     0.1000    0.0054
##    120        0.4969             nan     0.1000    0.0044
##    140        0.4438             nan     0.1000    0.0034
##    150        0.4210             nan     0.1000    0.0037
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1939
##      2        1.4836             nan     0.1000    0.1291
##      3        1.4000             nan     0.1000    0.1089
##      4        1.3296             nan     0.1000    0.0950
##      5        1.2701             nan     0.1000    0.0736
##      6        1.2241             nan     0.1000    0.0772
##      7        1.1757             nan     0.1000    0.0742
##      8        1.1307             nan     0.1000    0.0564
##      9        1.0954             nan     0.1000    0.0531
##     10        1.0623             nan     0.1000    0.0472
##     20        0.7978             nan     0.1000    0.0329
##     40        0.5170             nan     0.1000    0.0219
##     60        0.3561             nan     0.1000    0.0117
##     80        0.2607             nan     0.1000    0.0025
##    100        0.1951             nan     0.1000    0.0035
##    120        0.1467             nan     0.1000    0.0045
##    140        0.1126             nan     0.1000    0.0027
##    150        0.1006             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2443
##      2        1.4542             nan     0.1000    0.1684
##      3        1.3501             nan     0.1000    0.1379
##      4        1.2621             nan     0.1000    0.1247
##      5        1.1861             nan     0.1000    0.0964
##      6        1.1254             nan     0.1000    0.0838
##      7        1.0727             nan     0.1000    0.0726
##      8        1.0273             nan     0.1000    0.0777
##      9        0.9805             nan     0.1000    0.0754
##     10        0.9294             nan     0.1000    0.0567
##     20        0.6380             nan     0.1000    0.0348
##     40        0.3506             nan     0.1000    0.0133
##     60        0.2184             nan     0.1000    0.0089
##     80        0.1384             nan     0.1000    0.0053
##    100        0.0935             nan     0.1000    0.0024
##    120        0.0663             nan     0.1000    0.0011
##    140        0.0487             nan     0.1000    0.0010
##    150        0.0427             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1273
##      2        1.5216             nan     0.1000    0.0869
##      3        1.4632             nan     0.1000    0.0670
##      4        1.4184             nan     0.1000    0.0545
##      5        1.3827             nan     0.1000    0.0475
##      6        1.3516             nan     0.1000    0.0473
##      7        1.3220             nan     0.1000    0.0388
##      8        1.2958             nan     0.1000    0.0328
##      9        1.2749             nan     0.1000    0.0379
##     10        1.2500             nan     0.1000    0.0317
##     20        1.0779             nan     0.1000    0.0220
##     40        0.8786             nan     0.1000    0.0100
##     60        0.7436             nan     0.1000    0.0078
##     80        0.6421             nan     0.1000    0.0087
##    100        0.5633             nan     0.1000    0.0048
##    120        0.4987             nan     0.1000    0.0038
##    140        0.4470             nan     0.1000    0.0040
##    150        0.4236             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1890
##      2        1.4875             nan     0.1000    0.1369
##      3        1.3998             nan     0.1000    0.1107
##      4        1.3298             nan     0.1000    0.0857
##      5        1.2739             nan     0.1000    0.0727
##      6        1.2276             nan     0.1000    0.0740
##      7        1.1827             nan     0.1000    0.0569
##      8        1.1461             nan     0.1000    0.0697
##      9        1.1039             nan     0.1000    0.0583
##     10        1.0687             nan     0.1000    0.0520
##     20        0.8029             nan     0.1000    0.0329
##     40        0.5249             nan     0.1000    0.0178
##     60        0.3687             nan     0.1000    0.0102
##     80        0.2652             nan     0.1000    0.0071
##    100        0.1997             nan     0.1000    0.0032
##    120        0.1528             nan     0.1000    0.0032
##    140        0.1177             nan     0.1000    0.0027
##    150        0.1045             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2420
##      2        1.4568             nan     0.1000    0.1754
##      3        1.3460             nan     0.1000    0.1396
##      4        1.2585             nan     0.1000    0.1171
##      5        1.1863             nan     0.1000    0.0920
##      6        1.1265             nan     0.1000    0.0817
##      7        1.0756             nan     0.1000    0.0769
##      8        1.0264             nan     0.1000    0.0721
##      9        0.9819             nan     0.1000    0.0681
##     10        0.9406             nan     0.1000    0.0591
##     20        0.6316             nan     0.1000    0.0361
##     40        0.3418             nan     0.1000    0.0149
##     60        0.2127             nan     0.1000    0.0106
##     80        0.1368             nan     0.1000    0.0034
##    100        0.0952             nan     0.1000    0.0034
##    120        0.0684             nan     0.1000    0.0021
##    140        0.0501             nan     0.1000    0.0005
##    150        0.0439             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1320
##      2        1.5216             nan     0.1000    0.0843
##      3        1.4631             nan     0.1000    0.0680
##      4        1.4171             nan     0.1000    0.0545
##      5        1.3804             nan     0.1000    0.0518
##      6        1.3471             nan     0.1000    0.0452
##      7        1.3178             nan     0.1000    0.0420
##      8        1.2914             nan     0.1000    0.0369
##      9        1.2684             nan     0.1000    0.0386
##     10        1.2424             nan     0.1000    0.0310
##     20        1.0713             nan     0.1000    0.0209
##     40        0.8699             nan     0.1000    0.0127
##     60        0.7362             nan     0.1000    0.0099
##     80        0.6337             nan     0.1000    0.0050
##    100        0.5548             nan     0.1000    0.0054
##    120        0.4931             nan     0.1000    0.0037
##    140        0.4399             nan     0.1000    0.0034
##    150        0.4168             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1916
##      2        1.4837             nan     0.1000    0.1368
##      3        1.3943             nan     0.1000    0.1026
##      4        1.3284             nan     0.1000    0.0943
##      5        1.2688             nan     0.1000    0.0807
##      6        1.2172             nan     0.1000    0.0785
##      7        1.1688             nan     0.1000    0.0708
##      8        1.1253             nan     0.1000    0.0612
##      9        1.0868             nan     0.1000    0.0535
##     10        1.0532             nan     0.1000    0.0487
##     20        0.7898             nan     0.1000    0.0335
##     40        0.5137             nan     0.1000    0.0181
##     60        0.3527             nan     0.1000    0.0079
##     80        0.2603             nan     0.1000    0.0065
##    100        0.1946             nan     0.1000    0.0043
##    120        0.1517             nan     0.1000    0.0040
##    140        0.1166             nan     0.1000    0.0020
##    150        0.1024             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2428
##      2        1.4534             nan     0.1000    0.1680
##      3        1.3452             nan     0.1000    0.1410
##      4        1.2560             nan     0.1000    0.1093
##      5        1.1858             nan     0.1000    0.0969
##      6        1.1249             nan     0.1000    0.1004
##      7        1.0634             nan     0.1000    0.0733
##      8        1.0158             nan     0.1000    0.0748
##      9        0.9690             nan     0.1000    0.0665
##     10        0.9283             nan     0.1000    0.0550
##     20        0.6117             nan     0.1000    0.0315
##     40        0.3407             nan     0.1000    0.0174
##     60        0.2020             nan     0.1000    0.0066
##     80        0.1283             nan     0.1000    0.0051
##    100        0.0843             nan     0.1000    0.0023
##    120        0.0615             nan     0.1000    0.0014
##    140        0.0454             nan     0.1000    0.0006
##    150        0.0393             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1295
##      2        1.5230             nan     0.1000    0.0867
##      3        1.4647             nan     0.1000    0.0682
##      4        1.4193             nan     0.1000    0.0549
##      5        1.3828             nan     0.1000    0.0483
##      6        1.3512             nan     0.1000    0.0449
##      7        1.3223             nan     0.1000    0.0409
##      8        1.2958             nan     0.1000    0.0367
##      9        1.2710             nan     0.1000    0.0349
##     10        1.2495             nan     0.1000    0.0327
##     20        1.0748             nan     0.1000    0.0222
##     40        0.8759             nan     0.1000    0.0107
##     60        0.7410             nan     0.1000    0.0088
##     80        0.6352             nan     0.1000    0.0062
##    100        0.5559             nan     0.1000    0.0042
##    120        0.4950             nan     0.1000    0.0030
##    140        0.4457             nan     0.1000    0.0039
##    150        0.4225             nan     0.1000    0.0039
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1843
##      2        1.4874             nan     0.1000    0.1368
##      3        1.3992             nan     0.1000    0.1081
##      4        1.3282             nan     0.1000    0.0901
##      5        1.2715             nan     0.1000    0.0700
##      6        1.2264             nan     0.1000    0.0788
##      7        1.1773             nan     0.1000    0.0670
##      8        1.1349             nan     0.1000    0.0663
##      9        1.0956             nan     0.1000    0.0484
##     10        1.0658             nan     0.1000    0.0512
##     20        0.7949             nan     0.1000    0.0252
##     40        0.5184             nan     0.1000    0.0105
##     60        0.3629             nan     0.1000    0.0065
##     80        0.2643             nan     0.1000    0.0066
##    100        0.1986             nan     0.1000    0.0041
##    120        0.1534             nan     0.1000    0.0047
##    140        0.1187             nan     0.1000    0.0022
##    150        0.1064             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2425
##      2        1.4552             nan     0.1000    0.1700
##      3        1.3467             nan     0.1000    0.1398
##      4        1.2585             nan     0.1000    0.1086
##      5        1.1903             nan     0.1000    0.1020
##      6        1.1262             nan     0.1000    0.0898
##      7        1.0704             nan     0.1000    0.0787
##      8        1.0199             nan     0.1000    0.0813
##      9        0.9703             nan     0.1000    0.0740
##     10        0.9259             nan     0.1000    0.0602
##     20        0.6164             nan     0.1000    0.0401
##     40        0.3384             nan     0.1000    0.0173
##     60        0.2077             nan     0.1000    0.0067
##     80        0.1334             nan     0.1000    0.0034
##    100        0.0916             nan     0.1000    0.0032
##    120        0.0676             nan     0.1000    0.0012
##    140        0.0521             nan     0.1000    0.0010
##    150        0.0454             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1309
##      2        1.5221             nan     0.1000    0.0924
##      3        1.4615             nan     0.1000    0.0673
##      4        1.4168             nan     0.1000    0.0554
##      5        1.3807             nan     0.1000    0.0512
##      6        1.3482             nan     0.1000    0.0477
##      7        1.3180             nan     0.1000    0.0371
##      8        1.2938             nan     0.1000    0.0365
##      9        1.2714             nan     0.1000    0.0423
##     10        1.2441             nan     0.1000    0.0295
##     20        1.0718             nan     0.1000    0.0225
##     40        0.8706             nan     0.1000    0.0125
##     60        0.7394             nan     0.1000    0.0083
##     80        0.6384             nan     0.1000    0.0066
##    100        0.5598             nan     0.1000    0.0058
##    120        0.4991             nan     0.1000    0.0057
##    140        0.4451             nan     0.1000    0.0035
##    150        0.4220             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1965
##      2        1.4821             nan     0.1000    0.1394
##      3        1.3934             nan     0.1000    0.1069
##      4        1.3261             nan     0.1000    0.0910
##      5        1.2675             nan     0.1000    0.0736
##      6        1.2207             nan     0.1000    0.0725
##      7        1.1753             nan     0.1000    0.0729
##      8        1.1307             nan     0.1000    0.0583
##      9        1.0943             nan     0.1000    0.0550
##     10        1.0601             nan     0.1000    0.0550
##     20        0.7910             nan     0.1000    0.0304
##     40        0.5204             nan     0.1000    0.0221
##     60        0.3593             nan     0.1000    0.0070
##     80        0.2633             nan     0.1000    0.0062
##    100        0.1994             nan     0.1000    0.0051
##    120        0.1514             nan     0.1000    0.0028
##    140        0.1198             nan     0.1000    0.0013
##    150        0.1068             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2423
##      2        1.4542             nan     0.1000    0.1766
##      3        1.3433             nan     0.1000    0.1419
##      4        1.2537             nan     0.1000    0.1155
##      5        1.1820             nan     0.1000    0.0939
##      6        1.1233             nan     0.1000    0.0820
##      7        1.0712             nan     0.1000    0.0883
##      8        1.0170             nan     0.1000    0.0713
##      9        0.9731             nan     0.1000    0.0673
##     10        0.9316             nan     0.1000    0.0657
##     20        0.6226             nan     0.1000    0.0387
##     40        0.3332             nan     0.1000    0.0170
##     60        0.2077             nan     0.1000    0.0073
##     80        0.1387             nan     0.1000    0.0051
##    100        0.0963             nan     0.1000    0.0026
##    120        0.0702             nan     0.1000    0.0011
##    140        0.0529             nan     0.1000    0.0008
##    150        0.0466             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1280
##      2        1.5241             nan     0.1000    0.0901
##      3        1.4653             nan     0.1000    0.0661
##      4        1.4223             nan     0.1000    0.0560
##      5        1.3855             nan     0.1000    0.0508
##      6        1.3531             nan     0.1000    0.0423
##      7        1.3260             nan     0.1000    0.0402
##      8        1.3003             nan     0.1000    0.0368
##      9        1.2769             nan     0.1000    0.0386
##     10        1.2516             nan     0.1000    0.0307
##     20        1.0829             nan     0.1000    0.0206
##     40        0.8799             nan     0.1000    0.0130
##     60        0.7446             nan     0.1000    0.0099
##     80        0.6438             nan     0.1000    0.0081
##    100        0.5638             nan     0.1000    0.0052
##    120        0.4994             nan     0.1000    0.0051
##    140        0.4474             nan     0.1000    0.0040
##    150        0.4240             nan     0.1000    0.0032
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1827
##      2        1.4879             nan     0.1000    0.1361
##      3        1.4014             nan     0.1000    0.1095
##      4        1.3325             nan     0.1000    0.0830
##      5        1.2786             nan     0.1000    0.0858
##      6        1.2251             nan     0.1000    0.0676
##      7        1.1822             nan     0.1000    0.0692
##      8        1.1391             nan     0.1000    0.0666
##      9        1.0979             nan     0.1000    0.0603
##     10        1.0617             nan     0.1000    0.0456
##     20        0.8127             nan     0.1000    0.0388
##     40        0.5263             nan     0.1000    0.0183
##     60        0.3642             nan     0.1000    0.0097
##     80        0.2665             nan     0.1000    0.0051
##    100        0.2004             nan     0.1000    0.0051
##    120        0.1544             nan     0.1000    0.0041
##    140        0.1218             nan     0.1000    0.0012
##    150        0.1063             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2386
##      2        1.4572             nan     0.1000    0.1705
##      3        1.3492             nan     0.1000    0.1322
##      4        1.2668             nan     0.1000    0.1034
##      5        1.2000             nan     0.1000    0.1055
##      6        1.1359             nan     0.1000    0.0928
##      7        1.0785             nan     0.1000    0.0773
##      8        1.0300             nan     0.1000    0.0733
##      9        0.9861             nan     0.1000    0.0655
##     10        0.9446             nan     0.1000    0.0711
##     20        0.6139             nan     0.1000    0.0327
##     40        0.3405             nan     0.1000    0.0147
##     60        0.2109             nan     0.1000    0.0072
##     80        0.1379             nan     0.1000    0.0049
##    100        0.0934             nan     0.1000    0.0022
##    120        0.0677             nan     0.1000    0.0015
##    140        0.0503             nan     0.1000    0.0014
##    150        0.0432             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1340
##      2        1.5210             nan     0.1000    0.0885
##      3        1.4620             nan     0.1000    0.0696
##      4        1.4167             nan     0.1000    0.0563
##      5        1.3795             nan     0.1000    0.0470
##      6        1.3487             nan     0.1000    0.0462
##      7        1.3202             nan     0.1000    0.0422
##      8        1.2939             nan     0.1000    0.0404
##      9        1.2670             nan     0.1000    0.0340
##     10        1.2447             nan     0.1000    0.0316
##     20        1.0722             nan     0.1000    0.0201
##     40        0.8721             nan     0.1000    0.0112
##     60        0.7339             nan     0.1000    0.0080
##     80        0.6376             nan     0.1000    0.0085
##    100        0.5603             nan     0.1000    0.0055
##    120        0.4961             nan     0.1000    0.0040
##    140        0.4429             nan     0.1000    0.0029
##    150        0.4207             nan     0.1000    0.0026
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1974
##      2        1.4824             nan     0.1000    0.1346
##      3        1.3952             nan     0.1000    0.1104
##      4        1.3237             nan     0.1000    0.0933
##      5        1.2653             nan     0.1000    0.0773
##      6        1.2161             nan     0.1000    0.0651
##      7        1.1746             nan     0.1000    0.0638
##      8        1.1348             nan     0.1000    0.0526
##      9        1.1016             nan     0.1000    0.0572
##     10        1.0660             nan     0.1000    0.0567
##     20        0.8025             nan     0.1000    0.0362
##     40        0.5212             nan     0.1000    0.0173
##     60        0.3620             nan     0.1000    0.0125
##     80        0.2609             nan     0.1000    0.0067
##    100        0.1946             nan     0.1000    0.0042
##    120        0.1492             nan     0.1000    0.0030
##    140        0.1149             nan     0.1000    0.0020
##    150        0.1013             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2430
##      2        1.4514             nan     0.1000    0.1715
##      3        1.3429             nan     0.1000    0.1361
##      4        1.2576             nan     0.1000    0.1116
##      5        1.1870             nan     0.1000    0.1004
##      6        1.1242             nan     0.1000    0.0912
##      7        1.0668             nan     0.1000    0.0829
##      8        1.0176             nan     0.1000    0.0574
##      9        0.9797             nan     0.1000    0.0682
##     10        0.9368             nan     0.1000    0.0749
##     20        0.6107             nan     0.1000    0.0435
##     40        0.3382             nan     0.1000    0.0209
##     60        0.1974             nan     0.1000    0.0042
##     80        0.1272             nan     0.1000    0.0038
##    100        0.0859             nan     0.1000    0.0025
##    120        0.0610             nan     0.1000    0.0022
##    140        0.0443             nan     0.1000    0.0007
##    150        0.0384             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1322
##      2        1.5224             nan     0.1000    0.0857
##      3        1.4645             nan     0.1000    0.0687
##      4        1.4181             nan     0.1000    0.0563
##      5        1.3823             nan     0.1000    0.0522
##      6        1.3492             nan     0.1000    0.0451
##      7        1.3200             nan     0.1000    0.0437
##      8        1.2930             nan     0.1000    0.0422
##      9        1.2647             nan     0.1000    0.0330
##     10        1.2433             nan     0.1000    0.0333
##     20        1.0714             nan     0.1000    0.0239
##     40        0.8680             nan     0.1000    0.0108
##     60        0.7386             nan     0.1000    0.0095
##     80        0.6353             nan     0.1000    0.0080
##    100        0.5564             nan     0.1000    0.0048
##    120        0.4950             nan     0.1000    0.0038
##    140        0.4448             nan     0.1000    0.0038
##    150        0.4205             nan     0.1000    0.0033
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1945
##      2        1.4843             nan     0.1000    0.1334
##      3        1.3966             nan     0.1000    0.1054
##      4        1.3278             nan     0.1000    0.0943
##      5        1.2678             nan     0.1000    0.0754
##      6        1.2193             nan     0.1000    0.0760
##      7        1.1716             nan     0.1000    0.0654
##      8        1.1310             nan     0.1000    0.0595
##      9        1.0935             nan     0.1000    0.0668
##     10        1.0544             nan     0.1000    0.0553
##     20        0.8000             nan     0.1000    0.0293
##     40        0.5190             nan     0.1000    0.0168
##     60        0.3611             nan     0.1000    0.0082
##     80        0.2676             nan     0.1000    0.0056
##    100        0.1993             nan     0.1000    0.0051
##    120        0.1527             nan     0.1000    0.0020
##    140        0.1191             nan     0.1000    0.0013
##    150        0.1044             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2423
##      2        1.4543             nan     0.1000    0.1841
##      3        1.3401             nan     0.1000    0.1304
##      4        1.2585             nan     0.1000    0.1146
##      5        1.1872             nan     0.1000    0.1003
##      6        1.1247             nan     0.1000    0.0825
##      7        1.0720             nan     0.1000    0.0784
##      8        1.0226             nan     0.1000    0.0758
##      9        0.9753             nan     0.1000    0.0649
##     10        0.9360             nan     0.1000    0.0660
##     20        0.6223             nan     0.1000    0.0342
##     40        0.3320             nan     0.1000    0.0114
##     60        0.2013             nan     0.1000    0.0057
##     80        0.1312             nan     0.1000    0.0040
##    100        0.0891             nan     0.1000    0.0010
##    120        0.0645             nan     0.1000    0.0026
##    140        0.0481             nan     0.1000    0.0014
##    150        0.0406             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1289
##      2        1.5224             nan     0.1000    0.0863
##      3        1.4642             nan     0.1000    0.0692
##      4        1.4193             nan     0.1000    0.0527
##      5        1.3846             nan     0.1000    0.0519
##      6        1.3503             nan     0.1000    0.0461
##      7        1.3206             nan     0.1000    0.0390
##      8        1.2945             nan     0.1000    0.0362
##      9        1.2717             nan     0.1000    0.0380
##     10        1.2469             nan     0.1000    0.0328
##     20        1.0747             nan     0.1000    0.0203
##     40        0.8750             nan     0.1000    0.0119
##     60        0.7420             nan     0.1000    0.0091
##     80        0.6415             nan     0.1000    0.0055
##    100        0.5650             nan     0.1000    0.0061
##    120        0.5009             nan     0.1000    0.0038
##    140        0.4495             nan     0.1000    0.0031
##    150        0.4259             nan     0.1000    0.0040
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1875
##      2        1.4869             nan     0.1000    0.1367
##      3        1.3997             nan     0.1000    0.1094
##      4        1.3296             nan     0.1000    0.0910
##      5        1.2727             nan     0.1000    0.0791
##      6        1.2222             nan     0.1000    0.0752
##      7        1.1753             nan     0.1000    0.0626
##      8        1.1350             nan     0.1000    0.0680
##      9        1.0941             nan     0.1000    0.0533
##     10        1.0609             nan     0.1000    0.0479
##     20        0.7953             nan     0.1000    0.0258
##     40        0.5363             nan     0.1000    0.0178
##     60        0.3710             nan     0.1000    0.0136
##     80        0.2627             nan     0.1000    0.0077
##    100        0.1964             nan     0.1000    0.0041
##    120        0.1505             nan     0.1000    0.0039
##    140        0.1166             nan     0.1000    0.0021
##    150        0.1021             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2440
##      2        1.4548             nan     0.1000    0.1803
##      3        1.3460             nan     0.1000    0.1365
##      4        1.2607             nan     0.1000    0.1107
##      5        1.1902             nan     0.1000    0.0969
##      6        1.1291             nan     0.1000    0.0836
##      7        1.0761             nan     0.1000    0.0792
##      8        1.0268             nan     0.1000    0.0675
##      9        0.9838             nan     0.1000    0.0713
##     10        0.9409             nan     0.1000    0.0723
##     20        0.6312             nan     0.1000    0.0314
##     40        0.3422             nan     0.1000    0.0127
##     60        0.2070             nan     0.1000    0.0057
##     80        0.1332             nan     0.1000    0.0055
##    100        0.0923             nan     0.1000    0.0015
##    120        0.0670             nan     0.1000    0.0021
##    140        0.0490             nan     0.1000    0.0012
##    150        0.0417             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2427
##      2        1.4539             nan     0.1000    0.1718
##      3        1.3460             nan     0.1000    0.1435
##      4        1.2569             nan     0.1000    0.1017
##      5        1.1913             nan     0.1000    0.0985
##      6        1.1288             nan     0.1000    0.0862
##      7        1.0742             nan     0.1000    0.0836
##      8        1.0223             nan     0.1000    0.0770
##      9        0.9761             nan     0.1000    0.0771
##     10        0.9296             nan     0.1000    0.0699
##     20        0.6237             nan     0.1000    0.0321
##     40        0.3383             nan     0.1000    0.0144
##     60        0.2098             nan     0.1000    0.0101
##     80        0.1376             nan     0.1000    0.0034
##    100        0.0955             nan     0.1000    0.0034
##    120        0.0695             nan     0.1000    0.0009
##    140        0.0517             nan     0.1000    0.0011
##    150        0.0439             nan     0.1000    0.0007
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

Based on the results, random forest and boosted trees give a high accuracy and low out of sample error estimates. The random forest model showed the best accuracy rate 99.9% and so the lowest Out Of Sample error.  Since random forest has already given a very high accuracy, the models will not be combined for achieving a better accuracy, and this method will be used as the final model.

###<span style="color:blue"> Prediction Quiz Results </span>


```r
predict_final <- predict(model_rf, testing_final)
predict_final
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
