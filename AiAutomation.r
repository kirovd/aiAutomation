# Install and load necessary libraries
install.packages("dplyr")
install.packages("caret")
install.packages("xgboost")
install.packages("pROC")
install.packages("DMwR")
install.packages("ggplot2")
install.packages("reshape2")

library(dplyr)
library(caret)
library(xgboost)
library(pROC)
library(DMwR)      # For SMOTE
library(ggplot2)
library(reshape2)  # For melting the data frame

# Simulated Dataset Creation
n <- 1000
set.seed(123)  # for reproducibility
data <- data.frame(
  CustomerID = 1:n,
  Age = sample(18:70, n, replace = TRUE),
  Gender = sample(c("Male", "Female"), n, replace = TRUE),
  Balance = runif(n, min = 0, max = 100000),
  NumberOfProducts = sample(1:4, n, replace = TRUE),
  HasCreditCard = sample(c(0, 1), n, replace = TRUE),
  IsActiveMember = sample(c(0, 1), n, replace = TRUE),
  EstimatedSalary = runif(n, min = 10000, max = 100000),
  Churn = sample(c(0, 1), n, replace = TRUE)
)

# Convert factors
data$Gender <- as.factor(data$Gender)
data$HasCreditCard <- as.factor(data$HasCreditCard)
data$IsActiveMember <- as.factor(data$IsActiveMember)
data$Churn <- as.factor(data$Churn)

# Feature Engineering
data$InteractionFeature <- with(data, Age * Balance)

# Advanced EDA: Correlation plot
correlationPlot <- cor(data[sapply(data, is.numeric)])
correlationMelted <- reshape2::melt(correlationPlot)
ggplot2::ggplot(correlationMelted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Correlation Plot", x = "Variable 1", y = "Variable 2", fill = "Correlation")

# Handling Imbalanced Data: Applying SMOTE
balancedData <- DMwR::SMOTE(Churn ~ ., data = data, perc.over = 100, k = 5)

# Preparing Data for xgboost
labels <- as.numeric(as.factor(balancedData$Churn)) - 1
features <- as.matrix(balancedData[, -which(names(balancedData) == "Churn")])

# Splitting Data into Training and Test Sets
set.seed(123)
trainIndex <- caret::createDataPartition(labels, p = 0.8, list = FALSE)
trainData <- features[trainIndex,]
trainLabel <- labels[trainIndex]
testData <- features[-trainIndex,]
testLabel <- labels[-trainIndex]

# Training XGBoost Model
xgbModel <- xgboost::xgboost(data = trainData, label = trainLabel, nrounds = 100, objective = "binary:logistic")

# Model Evaluation
preds <- predict(xgbModel, testData)
rocCurve <- pROC::roc(testLabel, preds)
plot(rocCurve)

# Hyperparameter Tuning (Basic Example)
tunedXgbModel <- xgboost::xgboost(data = trainData, label = trainLabel, nrounds = 100, eta = 0.01, max_depth = 6, objective = "binary:logistic")

# Save the model
saveRDS(xgbModel, file = "advanced_churn_model.rds")
