# Prediction on Exercise
Ted Hwang  
September 19, 2016  

##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. I will use any of the other variables to predict with. I will create a report describing how I built my model, how I used cross validation, what I think the expected out of sample error is, and why I made the choices I did. I will also use your prediction model to predict 20 different test cases.

##Cleaning the Data
I'll first load the packages that we need and load the data into the R environment. 

```r
    library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```r
    setwd("/Users/Thwang/R\ Programming\ Class/8_ML/project")
    loc_file = './pml-training.csv'
    loc_file1='./pml-testing.csv'
    data = read.csv(loc_file,na.strings=c('NA','#DIV/0!',''))
    fin_testing = read.csv(loc_file1,na.strings=c('NA','#DIV/0',''))
    count_na = sum(is.na(data)==TRUE)
    total_entry = ncol(data)*nrow(data)
    percent_of_na = count_na/total_entry
    print(percent_of_na)
```

```
## [1] 0.6131835
```

As you can see, there are a lot of missing values. The machine learning algorithms in R doesn't like to deal with missing values. Therefore I'll have to remove the columns that have missing values. 


```r
    missing = sapply(data, function(x) {sum(is.na(x)==TRUE)})
    fin_missing = sapply(fin_testing,function(x) {sum(is.na(x)==TRUE)})
    
    good_names = names(which(missing==0))
    fin_good_names = names(which(fin_missing==0))
    
    c_data = data[,good_names]
    c_testing = fin_testing[,fin_good_names]
    
    no_name_data = grepl('X|timestamp|user_name',names(c_data))
    no_name_testing = grepl('X|timestamp|user_name',names(c_testing))
    
    c_data = c_data[,which(no_name_data==FALSE)]
    c_testing = c_testing[,which(no_name_testing==FALSE)]
    
    
    ncol(data)
```

```
## [1] 160
```

```r
    ncol(c_data)
```

```
## [1] 55
```

I removed 100 columns that held missing values from the dataset. Now, I can feed this data set into a machine learning algorithm.

##Cross Validation
I want to subdivide the training dataset into two parts. 80 percent of the dataset will be used to train the algorithm and the remaining 20 percent will be used to test the dataset. 

```r
    set.seed(517)
    t_index = createDataPartition(y=c_data$classe,p=.8,list=FALSE)
    training = c_data[t_index,]
    testing = c_data[-t_index,]
```

##Training the Algorithm
The best algorithm to use for this situation will be random forests since there are so many variables and the end result is a factor variable. I can use the caret package to train the training dataset to fit a random forest model.


```r
    set.seed(518)
    
    control = trainControl(method='cv',number=10)
    fit = train(classe~.,data=training,method='rf',trControl=control,
                ntree=200)
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

```r
    print(fit)
```

```
## Random Forest 
## 
## 15699 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 14129, 14128, 14129, 14128, 14130, 14129, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9946498  0.9932322
##   28    0.9978982  0.9973415
##   54    0.9947775  0.9933939
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```

The result looks promising. Using 28 trees, the algorithm was able to fit the training dataset up with an accuracy of 99.78%. We will use the testing dataset to see the model fit works outside of the training. 


```r
    pre_test = predict(fit,newdata = testing)
    confusionMatrix(pre_test,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  757    0    0    0
##          C    0    2  683    1    0
##          D    0    0    1  641    3
##          E    0    0    0    1  718
## 
## Overall Statistics
##                                          
##                Accuracy : 0.998          
##                  95% CI : (0.996, 0.9991)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9974         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9985   0.9969   0.9958
## Specificity            1.0000   1.0000   0.9991   0.9988   0.9997
## Pos Pred Value         1.0000   1.0000   0.9956   0.9938   0.9986
## Neg Pred Value         1.0000   0.9994   0.9997   0.9994   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1930   0.1741   0.1634   0.1830
## Detection Prevalence   0.2845   0.1930   0.1749   0.1644   0.1833
## Balanced Accuracy      1.0000   0.9987   0.9988   0.9978   0.9978
```

We see that the accuracy using the model on the testing set is 99.8%. This model appears to be really good. 


```r
    pred_answers = predict(fit, c_testing)
    print(pred_answers)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



##Conclusion
I used a random forest model to fit a training data set to get an accuracy of 99.78%. Then, used the model on the testing data set to get an accuracy of 99.8%. It appears that my model is very good. I will use this model on the actual testing dataset to get a prediction. 
