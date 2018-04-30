library(readr)
library(tidyverse)
library(randomForest)
library(caret)
library(MASS)
library(e1071)

emotion_table_ck <- read_csv("~/Documents/CSC481/Final Project/Project Code/emotionTable.txt")
features_ck <- read_csv("~/Documents/CSC481/Final Project/Project Code/features_ova.csv", 
                     col_names = FALSE)


face_data_ck <- data.frame(emotion_table_ck$emotion, features_ck)
names(face_data_ck)[1] <- 'emotion'

face_data_ck$emotion <- as.factor(face_data_ck$emotion)

rm(emotion_table_ck, features_ck)

set.seed(1)
train_control <- trainControl(method = "cv", number = 10, savePredictions = TRUE)

model_lda_ck <- train(emotion~., data = face_data_ck, trControl = train_control, method = 'lda')
model_rf_ck <- train(emotion~., data = face_data_ck, trControl = train_control, method = 'rf')
model_svm_ck <- train(emotion~., data = face_data_ck, trControl = train_control, method = 'svmLinear2')

pred_lda_ck <- model_lda_ck$pred
confusionMatrix(pred_lda_ck$pred, pred_lda_ck$obs)
model_rf_ck
model_svm_ck


# Testing how many eigenvectors are needed
accuracy <- rep(0,80)
for (i in 1:80){
  faceData_sub <- face_data_ck[,0:(1+i)]
  model_lda_loop <- train(emotion~., data = faceData_sub, trControl = train_control, method = 'lda')
  accuracy[i] <- model_lda_loop$results[2][[1]]
}

plot(accuracy, type = 'l', xlab = "Number of Eigenvectors", ylab = 'Accuracy', main = 'LDA Accuracy by Number
     of Eigenvectors')


# PCA plot
library(RColorBrewer)

levels(face_data_ck$emotion) <- c("Angry","Contempt","Disgust","Fear","Happy","Sadness","Surprise")
ggplot(face_data_ck, aes(X1, X2, color = emotion)) + geom_point(shape=13) + scale_color_brewer(palette = "Accent") +
  xlab("Principal Component 1") + ylab("Principal Component 2")

##### JAFFE dataset

emotion_table_j <- read_csv("~/Documents/CSC481/Final Project/Project Code/emotionTable_j.txt")
emotion_table_j$emotion <- as.factor(emotion_table_j$emotion)

features_j <- read_csv("~/Documents/CSC481/Final Project/Project Code/features_j.csv", 
                     col_names = FALSE)

face_data_j <- bind_cols(emotion_table_j, features_j)

rm(emotion_table_j, features_j)

# Cross Validation
set.seed(1)
train_control <- trainControl(method = "cv", number = 10, savePredictions = TRUE)

model_lda_j <- train(emotion~., data = face_data_j, trControl = train_control, method = 'lda')
model_rf_j <- train(emotion~., data = face_data_j, trControl = train_control, method = 'rf')
model_svm_j <- train(emotion~., data = face_data_j, trControl = train_control, method = 'svmLinear2')

model_lda_j
model_rf_j
model_svm_j

pred_lda_j <- model_lda_j$pred

confusionMatrix(pred_lda_j$pred, pred_lda_j$obs)

summary(faceData_j$emotion)

# Testing how many eigenvectors are needed
accuracy <- rep(0,100)
for (i in 1:100){
  faceData_j_sub <- faceData_j[,0:(1+i)]
  model_lda_loop <- train(emotion~., data = faceData_j_sub, trControl = train_control, method = 'lda')
  accuracy[i] <- model_lda_loop$results[2][[1]]
}

plot(accuracy, type = 'l', xlab = "Number of Eigenvectors", ylab = 'Accuracy', main = 'LDA Accuracy by Number
     of Eigenvectors')