data <- read.table("alldata.tsv", stringsAsFactors = FALSE,
                   header = TRUE)
testIDs <- read.csv("splits_F20.csv", header = TRUE)

split1 <- read.csv("split_1/test.tsv",stringsAsFactors = FALSE,
                   header = TRUE)
split <- read.table("project1_testIDs.dat")

myurl = "https://liangfgithub.github.io/MovieData/"
# use colClasses = 'NULL' to skip columns
ratings = read.csv(paste0(myurl, 'ratings.dat?raw=true'), 
                   sep = ':',
                   colClasses = c('integer', 'NULL'), 
                   header = FALSE)

#Randomly shuffle the data
ratings<-ratings[sample(nrow(ratings)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(ratings)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
for(i in 1:10){
  dir.create(paste("split_", i, sep=""))
  #Segement your data by fold using the which() function
  testIds <- which(folds==i,arr.ind=TRUE)
  test <- ratings[testIds, ]
  train <- ratings[-testIds, ]
  #Use the test and train data partitions however you desire...
  tmp_file_train <- paste("split_", i, "/", "train.tsv", sep="")
  write.table(train, file=tmp_file_train, 
              quote=TRUE, 
              row.names = FALSE,
              sep='\t')
  tmp_file_test <- paste("split_", i, "/", "test.tsv", sep="")
  write.table(test, file=tmp_file_test, 
              quote=TRUE, 
              row.names = FALSE,
              sep='\t')
}



for(j in 1:5){
  dir.create(paste("split_", j, sep=""))
  train <- data[-testIDs[,j], c("id", "sentiment", "review") ]
  test <- data[testIDs[,j], c("id", "review")]
  test.y <- data[testIDs[,j], c("id", "sentiment", "score")]
  
  tmp_file_name <- paste("split_", j, "/", "train.tsv", sep="")
  write.table(train, file=tmp_file_name, 
              quote=TRUE, 
              row.names = FALSE,
              sep='\t')
  tmp_file_name <- paste("split_", j, "/", "test.tsv", sep="")
  write.table(test, file=tmp_file_name, 
              quote=TRUE, 
              row.names = FALSE,
              sep='\t')
  tmp_file_name <- paste("split_", j, "/", "test_y.tsv", sep="")
  write.table(test.y, file=tmp_file_name, 
              quote=TRUE, 
              row.names = FALSE,
              sep='\t')
}

j = 5 # change value to switch between folder splits
setwd(paste("../split_", j, sep=""))
train = read.table("train.tsv",
                   stringsAsFactors = FALSE,
                   header = TRUE)
train$review = gsub('<.*?>', ' ', train$review)

library(text2vec)

stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "the", "us")
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
tmp.vocab = create_vocabulary(it_train, 
                              stopwords = stop_words, 
                              ngram = c(1L,4L))
tmp.vocab = prune_vocabulary(tmp.vocab, term_count_min = 10,
                             doc_proportion_max = 0.5,
                             doc_proportion_min = 0.001)
dtm_train  = create_dtm(it_train, vocab_vectorizer(tmp.vocab))

library(glmnet)
#set.seed()
tmpfit = glmnet(x = dtm_train, 
                y = train$sentiment, 
                alpha = 1,
                family='binomial')


myvocab = colnames(dtm_train)[which(tmpfit$beta[, 78] != 0)] #Vocabsize <2000 words 

#myvocab = colnames(dtm_train)[which(tmpfit$beta[, 65] != 0)] #Vocabsize <1000 words 


vectorizer = vocab_vectorizer(create_vocabulary(myvocab, 
                                               ngram = c(1L, 2L)))

dtm_train = create_dtm(it_train, vectorizer)


mylogit.cv = cv.glmnet(x = dtm_train, 
                       y = train$sentiment, 
                       alpha = 0,
                       family='binomial', 
                       type.measure = "auc")
mylogit.fit = glmnet(x = dtm_train, 
                     y = train$sentiment, 
                     alpha = 0,
                     lambda = mylogit.cv$lambda.min, 
                     family='binomial')

test = read.table("test.tsv",
                  stringsAsFactors = FALSE,
                  header = TRUE)
test$review <- gsub('<.*?>', ' ', test$review)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, vectorizer)
mypred = predict(mylogit.fit, dtm_test, type = "response")
output = data.frame(id = test$id, prob = as.vector(mypred))
write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep='\t')

test.y = read.table("test_y.tsv", header = TRUE)
pred = read.table("mysubmission.txt", header = TRUE)
pred = merge(pred, test.y, by="id")
library(pROC)
roc_obj = roc(pred$sentiment, pred$prob)
tmp = pROC::auc(roc_obj)
print(tmp)