# Project-2
library(dplyr)
library(ggplot2)
library(corrplot)
library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(cluster)


data<- read.csv("C:\\Users\\Lenovo\\OneDrive\\Desktop\\Assingments\\Analytics Practicum\\Project-2\\Software_Mailing_List.csv")
str(data)
summary(data)

# Checking for NA null values
sum(is.na(data))

# checking for Zero value 
sum(data$last_update_days_ago == 0, na.rm = T)

# placing numeric data into numeric variables for correlation
numeric_vars <- data %>% select_if(is.numeric)

#Checking zero for purchase and spending values, zero purchase equal to zero spending
sum(data$Purchase == 0 & data$Spending == 0, na.rm = T)
sum(data$sequence_number == 0, na.rm = T)

data_cluster <- data # copying data for clustering

# Convert binary variables into factors with descriptive labels for EDA
data$Purchase <- factor(data$Purchase, levels = c(0,1), labels = c("No", "Yes"))
data$US <- factor(data$US, levels = c(0,1), labels = c("Non-US", "US"))
data$Web.order <- factor(data$Web.order, levels = c(0,1), labels = c("No", "Yes"))
data$Gender.male <- factor(data$Gender.male, levels = c(0,1), labels = c("Female/Other", "Male"))
data$Address_is_res <- factor(data$Address_is_res, levels = c(0,1), labels = c("Business", "Residential"))

# Identify all columns that start with "source_"
source_cols <- grep("^source_", names(data), value = TRUE)

# Factor all source_* columns into "No" (0) and "Yes" (1)
for (col in source_cols) {
  data[[col]] <- factor(data[[col]], levels = c(0,1), labels = c("No", "Yes"))  }


# Check result
str(data[source_cols])

str(data)

# all the binary variables are factored.

# Spending distribution
ggplot(data, aes(x = Spending)) +
  geom_histogram(fill = "darkgreen", bins = 30) +
  labs(title = "Distribution of Spending", x = "Spending ($)", y = "Frequency")

# Transaction frequency distribution
ggplot(data, aes(x = Freq)) +
  geom_histogram(fill = "purple", bins = 30) +
  labs(title = "Distribution of Transaction Frequency", x = "Frequency", y = "Count")

# Boxplot: Spending by Purchase
ggplot(data, aes(x = Purchase, y = Spending, fill = Purchase)) +
  geom_boxplot() +
  labs(title = "Spending by Purchase", x = "Purchase", y = "Spending")

# Scatter: Frequency vs Spending
ggplot(data, aes(x = Freq, y = Spending)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "Relationship between Frequency and Spending",
       x = "Transaction Frequency", y = "Spending")



# Graph of sources 
source_counts <- sapply(source_cols, function(col) sum(data[[col]] == "Yes", na.rm = TRUE))

# Bar plot
barplot(source_counts,
        main = "Number of Customers by Source",
        xlab = "Sources",
        ylab = "Number of Customers",
        col = "skyblue",
        las = 2)  # rotate x-axis labels for readability



# For all source_* columns at once
source_cols <- grep("^source_", names(data), value = TRUE)

for (col in source_cols) {
  data[[col]] <- ifelse(as.character(data[[col]]) == "Yes", 1, 0) }


str(data) # Checking all variables are un-factored.

# Checking 15 source columns of each row contains only one value and rest all zero
# Count how many 1's in each row
row_sums <- rowSums(data[source_cols] == 1, na.rm = TRUE)

# Rows where rule is broken (more than one 1 or zero 1's)
violations <- data[row_sums > 1, ]   
nrow(violations)
head(violations)


#  Combine them into a single column named "source"
data$source <- apply(data[source_cols], 1, function(row) {
  # Getting the column names where value = 1
  sources_active <- source_cols[which(row == 1)]
  if (length(sources_active) == 0) {
    return("None")  
  } else {
    return(paste(sources_active, collapse = ", "))
  }
})

# Removing all sources from the data frame

data <- data[, !(names(data) %in% source_cols)]

str(data)



#Correlation (Numeric Only)

# Select numeric columns only

# Correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Correlation heatmap
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,tl.cex = 0.7,
         title = "Correlation Heatmap of Numeric Variables")

# Correlation matrix after removing all the sources

# placing numeric data into numeric variables for correlation, removing all the sources
numeric_vars_Nosource <- data %>% select_if(is.numeric)

cor_matrix_Nosource <- cor(numeric_vars_Nosource, use = "complete.obs")

# Correlation heatmap
corrplot(cor_matrix_Nosource, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,tl.cex = 0.7,
         title = "Correlation Heatmap of Numeric Variables No Source column")




# Chi-Square Test for target variable
#data$Gender.male <- as.factor(data$Gender.male)  # weak relation
#data$Purchase <- as.factor(data$Purchase)
chisq.test(table(data$Gender.male, data$Purchase))

#data$US <- as.factor(data$US) # weak relation
chisq.test(table(data$US, data$Purchase))

#data$Web.order <- as.factor(data$Web.order) # strong relation
chisq.test(table(data$Web.order, data$Purchase))

#data$Address_is_res <- as.factor(data$Address_is_res) # weak relation
chisq.test(table(data$Address_is_res, data$Purchase))

data$source <- as.factor(data$source) # Strong relation
chisq.test(table(data$source, data$Purchase))


# T-Test to find out the relation
t.test(Freq ~ Purchase, data = data)
ggplot(data, aes(x = factor(Purchase), y = Freq)) +
  geom_boxplot(fill = "lightgreen") +
  labs(x = "Purchase", y = "Frequency", title = "Frequency vs Purchase")


t.test(last_update_days_ago ~ Purchase, data = data)
ggplot(data, aes(x = factor(Purchase), y = last_update_days_ago)) +
  geom_boxplot(fill = "lightgreen") +
  labs(x = "Purchase", y = "last_update_days_ago", title = "last_update_days_ago vs Purchase")


t.test(Spending ~ Purchase, data = data)
ggplot(data, aes(x = factor(Purchase), y = Spending)) +
  geom_boxplot(fill = "lightgreen") +
  labs(x = "Purchase", y = "Spending", title = "Spending vs Purchase")


# data partition
set.seed(123)
split = sample.split(data$Purchase, SplitRatio = 0.8)
train_set = subset(data, split== TRUE)
test_set = subset(data, split == FALSE)

# spending posses the same properties of purchase, spending is removed
train_set$Spending <- NULL
test_set$Spending  <- NULL

#*********************************************************
# Decision tree

tree_model <- rpart(Purchase ~ ., data = train_set, method = "class",
                    control = rpart.control(cp = 0.001))

# Find best cp value (minimum xerror)
printcp(tree_model)
best_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]

# Prune the tree using best cp
pruned_tree <- prune(tree_model, cp = best_cp)

# Plot pruned tree
rpart.plot(pruned_tree, type = 2, extra = 104, fallen.leaves = TRUE,
           main = "Pruned Decision Tree")

# Predict on test data
pred <- predict(pruned_tree, newdata = test_set, type = "class")


# Confusion matrix 
confusionMatrix(pred, test_set$Purchase, positive = "Yes")

#************************************************

# Random forest

rf_model <- randomForest(Purchase ~ ., data = train_set, ntree = 100, importance = TRUE)

# Predict on test data
pred_rf <- predict(rf_model, newdata = test_set)

# Confusion matrix
confusionMatrix(pred_rf, test_set$Purchase, positive = "Yes")


# View variable importance
importance(rf_model)
varImpPlot(rf_model)

#*******************************************************
# --- Hierarchical clustering 


#Normalizing the spending column 
data_cluster$Spending <- (data_cluster$Spending - min(data_cluster$Spending)) / (max(data_cluster$Spending) - min(data_cluster$Spending))

data_cluster$last_update_days_ago <- (data_cluster$last_update_days_ago - min(data_cluster$last_update_days_ago)) / (max(data_cluster$last_update_days_ago) - min(data_cluster$last_update_days_ago))

data_cluster$X1st_update_days_ago <- (data_cluster$X1st_update_days_ago - min(data_cluster$X1st_update_days_ago)) / (max(data_cluster$X1st_update_days_ago) - min(data_cluster$X1st_update_days_ago))

head(data_cluster$Spending) # checking the values
data_cluster$sequence_number <- NULL 
str(data_cluster)


# 1) Make selected columns factors (adjust names as needed)
factor_cols <- c("US","Web.order","Gender.male","Address_is_res","Purchase",
                 grep("^source_", names(data_cluster), value = TRUE))
factor_cols <- intersect(factor_cols, names(data_cluster))
data_cluster[factor_cols] <- lapply(data_cluster[factor_cols], factor)

# 2) Gower distance (handles mixed types)
set.seed(123)
gower_dist <- daisy(data_cluster, metric = "gower")

# 3) Hierarchical clustering
hc <- hclust(gower_dist, method = "complete")

# 4) Choose k via average silhouette
ks <- 2:15
avg_sil <- sapply(ks, function(k) {
  cl <- cutree(hc, k = k)
  mean(silhouette(cl, gower_dist)[, 3])
})
k_opt <- ks[which.max(avg_sil)]
print(paste("Best k by avg silhouette:", k_opt))


# 5) Final cut + quick profiling
clusters <- cutree(hc, k = k_opt)
data_cluster$cluster_hc <- clusters
print("Cluster sizes:\n"); print(table(data_cluster$cluster_hc))

# Profile numeric variables by median
num_cols <- names(data_cluster)[sapply(data_cluster, is.numeric)]
if (length(num_cols) > 0) {
  prof <- aggregate(data_cluster[num_cols], list(cluster = clusters), median, na.rm = TRUE)
  print(prof)
}

# 6) Plots
plot(hc, main = "Hierarchical Clustering (Gower)", xlab = "", sub = "")
plot(ks, avg_sil, type = "b", xlab = "k", ylab = "Avg silhouette",
     main = "Silhouette vs k (Gower)")

#*****************************************************************
# K means clustering

data_num <- numeric_vars_Nosource # assigning numeric data

# Normalization function (min-max scaling)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization to the selected columns
data_num$last_update_days_ago <- normalize(data_num$last_update_days_ago)
data_num$X1st_update_days_ago <- normalize(data_num$X1st_update_days_ago)
data_num$Spending <- normalize(data_num$Spending)

# Pick k with a quick elbow plot 
wss <- sapply(1:10, function(k) {
  kmeans(data_num, centers = k, nstart = 50)$tot.withinss
})

plot(1:10, wss, type = "b",
     xlab = "k (number of clusters)", ylab = "Total within-cluster SS",
     main = "Elbow method")

# Fit k-means 
km <- kmeans(data_num, centers = 3, nstart = 50)

# Inspect results
km$centers            
km$size               
head(km$cluster)      

# Computing silhouette width for performance
sil <- silhouette(km$cluster, dist(data_num))
mean_silhouette <- mean(sil[, 3])   # Average silhouette score
mean_silhouette

#****************************
#adjusting probability
# Predicting probability (
pred_prob <- predict(rf_model, newdata = test_set, type = "prob")[, "Yes"]

# Adjusting predicted probabilities to reflect the true purchase rate
adjusted_prob <- pred_prob * (5.3 / 50)   # equivalent to multiplying by 0.106

# Add adjusted probabilities to test set
test_set$Adjusted_Prob <- adjusted_prob

# View first few adjusted probabilities
head(test_set$Adjusted_Prob)

