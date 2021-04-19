---
title: "Barbell Lifts"
author: "Asgeir Brenne"
date: "4/10/2021"
output:
  html_document:
    keep_md: yes
---



## Introduction

One thing that people regularly do is quantify how  much of a particular activity they do, but they rarely quantify how well they do it. In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The goal of the project is to predict the manner in which they did the exercise.

The project will follow these main steps:

* Data loading and cleaning (the training and test dataset files must be downloaded to the working directory)
* Basic exploratory analysis to understand the data
* Data preprocessing
* Machine learning model strategy, including methods and cross validation
* Training and testing

The ```caret``` and ```plyr``` packages are used in the process.



## Data Loading and Cleaning

We first load the training and testing data sets.


```r
training_raw <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv", na.strings = c("NA","#DIV/0!",""))
```

A quick look at the testing data set to identify the available data to predict on, allows us to reduce the list of potential predictors (many all-NA fields in testing set).


```r
all_NA <- sapply(names(testing), function(x) all(is.na(testing[,x])==TRUE)) ## All NA columns in test set
useCols <- names(all_NA)[all_NA==FALSE]
useCols <- useCols[-(1:7)] ## Only identifiers
useCols <- useCols[1:(length(useCols)-1)] ## Remove "problem_id" field
useCols <- c(useCols, "classe") ## FUll list of the columns to use in training set
```

So when we start modeling, the following columns will be available:

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```


## Basic exploratory analysis

Let's have a quick look at the data. We know that our aim is to predict the ```classe``` field, so let's first get a basic overview of that field. Also, despite having eliminated NZV columns, let's get a feel for the completeness of the data.


```r
count(training_raw, "classe")
```

```
##   classe freq
## 1      A 5580
## 2      B 3797
## 3      C 3422
## 4      D 3216
## 5      E 3607
```

```r
mean(is.na(training_raw[,useCols]))
```

```
## [1] 0
```

So we notice that there are no NA's in the columns we will use for the ML algorithms, and we have a fairly even distribution between the different classes we shall predict.


## Data preprocessing and controls

Let's partition the training set into ```training``` and ```validation``` partitions, in order to have a way to estimate out-of-sample errors after training the model.


```r
## Convert classe field to factor variable
training_raw$classe <- as.factor(training_raw$classe)

## Partitioning
inTrain <- createDataPartition(training_raw$classe, p=0.8, list=FALSE)
training <- training_raw[inTrain,useCols]
validation <- training_raw[-inTrain,useCols]

## Controls
trCont <- trainControl(method = "repeatedcv", number=5, repeats = 5)
set.seed(0)
```

So we have defined a common set of controls for cross-validation, using 5 folds and 5 repeats for each of the ML algorithms.


## Machine learning models

For classification problems such as this, decision trees k-nearest neighbors and random forests are typically good algorithms. Let's run  all so we later can compare the results


```r
## Basic decision tree
modTree <- train(classe ~ ., method="rpart", data = training, preProcess = c("center", "scale"), trControl = trCont)

## k-nearest neighbors
modKNN <- train(classe ~ ., method="knn", data = training, preProcess = c("center", "scale"), trControl = trCont)

## Random forest
modRF <- train(classe ~ ., method="rf", data = training, preProcess = c("center", "scale"), trControl = trCont)
```


## Model comparison and predicting on the test set

Let's compare the results of the model, on the basis of the ```accuracy``` measurement generated by using the ```confusionMatrix``` function.


```r
## Predictions
predTree <- predict(modTree, validation)
predKNN <- predict(modKNN, validation)
predRF <- predict(modRF, validation)

## Accuracies
accTree <- confusionMatrix(predTree, validation$classe)$overall[1]
accKNN <- confusionMatrix(predKNN, validation$classe)$overall[1]
accRF <- confusionMatrix(predRF, validation$classe)$overall[1]
data.frame(Model = c("Tree", "K-nearest", "Random forecast"), Accuracy = c(accTree,accKNN,accRF))
```

```
##             Model  Accuracy
## 1            Tree 0.4876370
## 2       K-nearest 0.9701759
## 3 Random forecast 0.9946470
```

So we see that (as expected), K-nearest neightbors and random forests outperform the simpler decision tree. With an accuracy of 99.4%, we expect an out-of-sample error of 0.6% when using the random forest model to predict on the test set.

The confusion matrix for the predictions based on the random forest model is shown below.

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    3    0    0    0
##          B    0  755    3    0    0
##          C    0    1  679   10    0
##          D    0    0    2  632    1
##          E    0    0    0    1  720
```

Ensembling the different models for a majority voting system, makes little sense given the accuracy we obtain using just one model. So finally, let's predict the classe variable on the test set using the random forest model. 


```r
data.frame(problem_id = testing$problem_id, Prediction = predict(modRF, testing))
```

```
##    problem_id Prediction
## 1           1          B
## 2           2          A
## 3           3          B
## 4           4          A
## 5           5          A
## 6           6          E
## 7           7          D
## 8           8          B
## 9           9          A
## 10         10          A
## 11         11          B
## 12         12          C
## 13         13          B
## 14         14          A
## 15         15          E
## 16         16          E
## 17         17          A
## 18         18          B
## 19         19          B
## 20         20          B
```


## Conclusion

Having looked at the available fields to predict upon in the ```testing``` data set, we are able to build a very accurate prediction model on the ```training``` data set. Partitioning out a ```validation``` set allowed us to evaluate the different models properly and produce a good estimate for out-of-sample error.

Both K-nearest neighbors and random forests, which normally are good to solve classification problems, worked very well.
