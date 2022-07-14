# Importing the dataset
df = read.csv('C:/Users/manis/Social_Network_Ads.csv')
View(df)
head(df)

# removing "userId" column which is not required
dataset = df[2:5]
View(dataset)
head(dataset)

summary(dataset)
str(dataset)

# label Encoding the Gender feature(dataset) 
dataset$Gender <- ifelse(dataset$Gender == "Male",1,0)
table(dataset$Gender)

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

table(dataset$Purchased)

str(dataset)


# pair plots of dataset
pairs(dataset)

# 

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# applying dataset without scaling the data
library(caret)
library(klaR)
# Fitting Logistic Regression to the Training set
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-4])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set[, 4], y_pred > 0.5)
cm


# Fitting Decision Tree Classification to the Training set
# install.packages('rpart')
library(rpart)
classifier = rpart(formula = Purchased ~ .,
                   data = training_set)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-4], type = 'class')

# Making the Confusion Matrix
confusionMatrix( test_set[, 4],y_pred)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-4],
                          y = training_set$Purchased,
                          ntree = 500)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-4])

# Making the Confusion Matrix
confusionMatrix( test_set[, 4],y_pred)

# Fitting Naive Bayes to the Training set
# install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-4],
                        y = training_set$Purchased)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-4])

# Making the Confusion Matrix
confusionMatrix( test_set[, 4],y_pred)

# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
y_pred = knn(train = training_set[, -4],
             test = test_set[, -4],
             cl = training_set[, 4],
             k = 5,
             prob = TRUE)


# Making the Confusion Matrix
confusionMatrix( test_set[, 4],y_pred)

# Fitting Kernel SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-4])

# Making the Confusion Matrix
confusionMatrix( test_set[, 4],y_pred)

# the best model for "Social_Network_Ads.csv" dataset is SVM classifier because of
# best accuracy score and other values when compare to other models. And the model is 
# run of all features and without scaling the data of dataset.


#Confusion Matrix and Statistics of SVM model:

"
Reference
Prediction  0  1
          0 58  6
          1  5 31

Accuracy : 0.89            
95% CI : (0.8117, 0.9438)
No Information Rate : 0.63            
P-Value [Acc > NIR] : 4.377e-09       

Kappa : 0.7627          

Mcnemar's Test P-Value : 1               
                                          
            Sensitivity : 0.9206          
            Specificity : 0.8378          
         Pos Pred Value : 0.9062          
         Neg Pred Value : 0.8611          
             Prevalence : 0.6300          
         Detection Rate : 0.5800          
   Detection Prevalence : 0.6400          
      Balanced Accuracy : 0.8792          
                                          
       'Positive' Class : 0   

"
