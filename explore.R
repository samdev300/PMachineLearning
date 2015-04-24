# explore.R

library(rpart)
library(rpart.plot)
library(randomForest)

training <- read.csv("data/pml-training.csv")
testing <- read.csv("data/pml-testing.csv")

# Briefly review the summary of the dataset - each variable
summary(training)
summary(testing)

# There are lots of missing data, so we'll need to take care of them, first.

# Since we'll need to predict the "classe" for each observation in testing set, 
# we need to make sure we're using the variables that exist in the testing set.
# We can't use the predictors that do not exist in the testing set.
# So remove all the columns in the testing set with all NA's.

# Get column names that have zero NA's.  This is the possible pool of predictors we could use.
colsForTesting <- colnames(testing)[colSums(is.na(testing)) == 0]

# Now get the column names for training set;  Include the "classe" response variable for training.
colsForTraining <- c(colsForTesting[1:59], "classe")


# Response variable: classe
# A = Exactly according to the specification
# B = Throwing the elbows to the front
# C = Lifting the dumbbell only halfway
# D = Lowering the dumbbell only halfway
# E = Throwing the hips to the front

table(training$classe)
# A    B    C    D    E 
# 5580 3797 3422 3216 3607 

prop.table(table(training$classe))
# See the proportion of each
# A         B         C         D         E 
# 0.2843747 0.1935073 0.1743961 0.1638977 0.1838243 




# Experiment with smaller set
training1 <- training[, colsForTraining]
summary(training1)
training1$X <- NULL

# Let's explore with smaller set of data 
set.seed(300)
spl <- sample.split(training1$classe, SplitRatio=.30)
tempPlaySet <- subset(training1, spl == TRUE)

# Split the tempPlaySet1 to train & test.
set.seed(300)
spl <- sample.split(tempPlaySet$classe, SplitRatio=.70)
miniTrain1 <- subset(tempPlaySet, spl == TRUE)
miniTest1 <- subset(tempPlaySet, spl == FALSE)

str(miniTrain1)

# Try CART model to get an idea of the significant predictors
miniModFit1 <- rpart(classe ~., data=miniTrain1, method="class")
prp(miniModFit1)

miniPred1 <- predict(miniModFit1, miniTest1, type="class")

table(miniPred1, miniTest1$classe)

miniAccuracy1 <- (487 + 282 + 279 + 225 + 263) / nrow(miniTest1)
miniAccuracy1  # = 0.8697622


# Now let's try Random Forest
miniModFitRF1 <- randomForest(classe ~., data=miniTrain1)

miniPredRF1 <- predict(miniModFitRF1, miniTest1)

table(miniPredRF1, miniTest1$classe)

miniAccuracyRF1 <- (502 + 340 + 303 + 287 + 320) / nrow(miniTest1)
miniAccuracyRF1 # = 0.9920725


# Now let's try Random Forest by caret package
miniModFitCRF1 <- train(classe ~., data=miniTrain1, method="rf")

miniPredCRF1 <- predict(miniModFitCRF1, miniTest1)

confusionMatrix(miniPredCRF1, miniTest1$classe)

# Accuracy = 0.9921

library(doMC)
registerDoMC(cores = 3)

# K-fold cross validation
fitControl <- trainControl(method="cv", number=10)

# Try it again with K-fold cross validation & multi-core processing
miniModFitCRF1b <- train(classe ~., data=miniTrain1, method="rf", trControl=fitControl)
# Took about 4 minutes

miniPredCRF1b <- predict(miniModFitCRF1b, miniTest1)
confusionMatrix(miniPredCRF1b, miniTest1$classe)
# Accuracy = 0.9909


ggplot(data=miniTrain1, aes(x=roll_belt, y=pitch_belt, color=classe)) + geom_point()


# ------------------------------------------------------------------------------------
# Let's build a real model using all the full data set

# Subset the only the necessary columns
training <- training[, colsForTraining]
training$X <- NULL

# Split the data into train & test set
set.seed(300)
spl <- sample.split(training$classe, SplitRatio=.70)
mTrain <- subset(training, spl == TRUE)
mTest <- subset(training, spl == FALSE)

# Apply Random Forest using caret package;  Apply k-fold cross valiation;

# K-fold cross validation
fitControl <- trainControl(method="cv", number=10)

# Try it again with K-fold cross validation & multi-core processing
set.seed(300)
modFitRF <- train(classe ~., data=mTrain, method="rf", trControl=fitControl)
# Took about 19 minutes

predRF <- predict(modFitRF, mTest)

confusionMatrix(predRF, mTest$classe)

# Accuracy = 0.9992


# Predict for the testing set
predTesting <- predict(modFitRF, testing)

predTesting


# --------------------------------------------------------------------------------
# Write 20 text files for submission

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("data/problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(predTesting)




# ------------------------------------------------------------------------------------------
# Citing
#
# Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition 
# of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI 
# (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
# 
# DocumentoDocumento [http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf]

http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201

# Read more: http://groupware.les.inf.puc-rio.br/har#wle_paper_section#ixzz3Y4CzylR6

