# Sentiment-Analysis-with-R

## Problem statement: 
The goal of this project is to build a binary classification model to
predict the sentiment of IMDB movie reviews. We were provided with a dataset where each
review is labelled as positive or negative.

## Objective: 

The evaluation metric is to determine AUC (area under curve) on the test
subset of the overall dataset. The assignment’s performance goal requires we achieve a minimal
AUC value of 0.96 over the five test datasets.

## Input and output of your model: 

The given movie review dataset file (alldata.tsv) has 50,000 rows (reviews) and 3 columns. Column 1, “new_id,” is the ID for each review (same as the row number). Column 2, “sentiment,” is the binary response. Column 3 is the review itself.
The provided csv file, splits_F20.csv, is used to split the 50,000 reviews into five sets of training
and test data subsets. For each of the five expected subsets, there is a column consisting of
25,000 “new_id” (or equivalently the row ID) values from the initial dataset. For model
interpretability and get full marks, the vocabulary size used for prediction should be less than
1000 ngrams.
