# Question 1
        svm_linear   svm_rbf      rf       knn1     knn5     knn10
err    0.026576    0.008177    0.016184  0.00954  0.011925  0.01448

## a    According to the error rates SVM Radial Basis Function out performs all the other classifiers.
## b    Using SVM Linear separates  the data linearly into two class but our data cannot be separated linearly also 8s and 9s in some cases looks same which means classes are overlapping so SVM RBF performs better than SVM linear. Random Forest creates a confusion matrix which creates a bunch of decision in the training model and according to that it will predict if the sample is in class A or in class B but our dataset being big and a lot of the samples hard to predict because of class overlap. So SVM RBF performs better than Random Forest.Knn 1 the  error rates are very low and its not reasonable because the machine is overfitting that is it is learned too much of just one class. So the machine is predicting sample 8 as class 9, hence  k=1 is not reliable. Where as knn5 is more reliable because the chances of ovefitting is comparatively less and in the knn classifier models k=10 is the best and most reliable model because the chances of overfitting is the least when compared to k=1 and k=5. Even though knn10 is most reliable but SVM RBF is performing better because SVM RBF can more precisely separate 8s and 9s than knn10.

## c.   Knn compare to assignment 1, at the lower values of k the model is performing way better and as we increase the value of k the error rate increase which is the same patter as assignment 1. This is because at low values of k the model is performing outstanding errors because while when we are training the model, the model is learning too much of class A so when we are giving a sample data to test and the model is predicting the  sample accurate but another sample which the model is supposed to predict in class B, it is still predicting as class A. so value of k very low is not a reliable value.


# Question 2
 
## Data Discription- The data consist of behaviour of people and corresponding risk to develop cancer. The data has 19 attributes, meaning 19 behaviour and 76 instances meaning 76 samples. The data is used for classification and which behaviour has the highest risk of having cancer. The count of group of interests are 21 and the count of group not of interests are 51.

## AUC Values
|    | Feature                    |   AUC |
|---:|:---------------------------|------:|
| 17 | empowerment_abilities      | 0.17  |
| 16 | empowerment_knowledge      | 0.195 |
| 10 | perception_severity        | 0.213 |
| 18 | empowerment_desires        | 0.226 |
|  9 | perception_vulnerability   | 0.248 |
|  8 | norm_fulfillment           | 0.25  |
| 13 | socialSupport_emotionality | 0.254 |
| 12 | motivation_willingness     | 0.267 |
|  2 | behavior_personalHygine    | 0.282 |
|  3 | intention_aggregation      | 0.298 |



## Question 3

        svm_linear     svm_rbf        rf        knn1      knn5       knn10
err    0.106667          0.12         0.16    0.146667  0.146667     0.16

## a.  For my data SVM LINEAR is performing better than any other model where as for for question 1 SVM RBF was the best performing.

## b.  In question 1 SVM RBF was the best performing because the two classes in data for Question 1 are overlapping. Which means that some 8s and 9s where hard to differentiate so they were not able to separate linearly because it end up mixing some samples from class 8s to 9s and the other way. But in my data the two classes are well separated and no overlapping is happening. So SVM linear is performing better in my data set because my data is linearly separable and also the model can find a nice support vector. Therefore SVM Linear is performing better in my data set compared to SVM RBF in question1.
