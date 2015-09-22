## Kaggle Scripts: Ponpare Coupon Purchase Prediction ###

# Setting the working directory
if (!exists(x = "path")) {
  path <- paste0(getwd(),"/kaggle-coupon-purchase-prediction")
}
setwd(path)

#add functions
normalit <- function(m){
  (m - min(m))/(max(m) - min(m))
}


# Read in all the input data
coupon_detail_train <- read.csv("Data/coupon_detail_train_en.csv")
coupon_list_train <- read.csv("Data/coupon_list_train_en.csv")
coupon_list_test <- read.csv("Data/coupon_list_test_en.csv")
user_list <- read.csv("Data/user_list_en.csv")

# Making of the train set
train <- merge(coupon_detail_train, coupon_list_train, by = "COUPON_ID_hash")
train <- merge(train, user_list, by = "USER_ID_hash")
train <- train[,c("COUPON_ID_hash","USER_ID_hash",
                  "en_GENRE_NAME","DISCOUNT_PRICE","PRICE_RATE",
                  "USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU",
                  "USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY",
                  "USABLE_DATE_BEFORE_HOLIDAY","en_ken_name","en_small_area_name")]

# Combine the test set with the train
coupon_list_test$USER_ID_hash <- "dummyuser"
cpchar <- coupon_list_test[,c("COUPON_ID_hash","USER_ID_hash",
                   "en_GENRE_NAME","DISCOUNT_PRICE","PRICE_RATE",
                   "USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU",
                   "USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY",
                   "USABLE_DATE_BEFORE_HOLIDAY","en_ken_name","en_small_area_name")]

train <- rbind(train,cpchar)

# NA imputation
train[is.na(train)] <- 1

# Feature engineering -- this appears to manipulate variables to change their distributions (How will this improve results??)
train$DISCOUNT_PRICE <- 1/log10(train$DISCOUNT_PRICE)
train$PRICE_RATE <- (train$PRICE_RATE*train$PRICE_RATE)/(100*100)

# Convert the factors to columns of 0's and 1's -- all categorical variables must be converted to dummy var first
train <- cbind(train[,c(1,2)],model.matrix(~ -1 + .,train[,-c(1,2)],
                                           contrasts.arg=lapply(train[,names(which(sapply(train[,-c(1,2)], is.factor)==TRUE))], contrasts, contrasts=FALSE)))


## ADD our own features in here.....



# Separate the test from train
test <- train[train$USER_ID_hash=="dummyuser",]
test <- test[,-2] ## this removed the USER_ID_hash field from the test data
train <- train[train$USER_ID_hash!="dummyuser",]

# Data frame of user characteristics -- individual user vectors are compared to the anon users in the test set to find the 'most similar' and pull out the Coupons purchased by the 'most similar'
uchar <- aggregate(.~USER_ID_hash, data=train[,-1],FUN=mean)
uchar$DISCOUNT_PRICE <- 1 ## why do this??
uchar$PRICE_RATE <- 1 ## why do this??

# Weight Matrix: GENRE_NAME DISCOUNT_PRICE PRICE_RATE USABLE_DATE_ ken_name small_area_name
# This creates a diagonal matrix that adds or reduces the weight/emphasis applied to each variable in the set.
# We need to add to this if we create our own features and we should experiment with applying different weights
require(Matrix)
W <- as.matrix(Diagonal(x=c(rep(3,13), rep(1,1), rep(0.2,1), rep(0,9), rep(1,47), rep(1,55))))


# Calculation of cosine similairties of users and coupons -- high score means user from train set is very similar to user who purchased coupon X in test set. 
# I think the implicit assumption here is that the two matrices have already been normalised. Therefore to get the cosine similarity scores we just find the Matrix product?
user_matrix <- as.matrix(uchar[,2:ncol(uchar)])
user_matrix <- apply(user_matrix, 2, normalit)
test_matrix <- as.matrix(test[,2:ncol(test)])
test_matrix <- apply(test_matrix, 2, normalit) ## this introduces NaN since some cols are all 0. need to update function to exlude all 0 cols.
score = user_matrix %*% W %*% t(test_matrix)


# Order the list of coupons according to similairties and take only first 10 coupons
uchar$PURCHASED_COUPONS <- do.call(rbind, lapply(1:nrow(uchar),FUN=function(i){
  purchased_cp <- paste(test$COUPON_ID_hash[order(score[i,], decreasing = TRUE)][1:10],collapse=" ")
  return(purchased_cp)
}))


# Make submission
submission <- merge(ulist, uchar, all.x=TRUE)
submission <- submission[,c("USER_ID_hash","PURCHASED_COUPONS")]
write.csv(submission, file="cosine_sim.csv", row.names=FALSE)