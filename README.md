## Distinguishing between Two Decision Trees Using Recursive Feature Engineering


To note, the file called Model_590_Cart is not something I would normally do. An assignment I was working on called for certain types of decision trees.

The data is provided by Kaggle and is part of the ongoing Boston Housing competition.

## Model 590 Cart


I performed recursive feature engineering to determine which variables provided the highest gini purity using Random Forest.

I then used those variables in a decision tree.

The document Model_590_cart.py satisfies and assignment that I was working on. However, to actually model this problem correctly I will explore other options.

## Data Exploration

Please read through my notes in the data exploration. This was a quick walk through of the data with some feature engineering with correlation feature engineering.

## Keras Example

In this file I am modeling the data with  simple MLP network using Keras and tensorflow. The model did not perform that well but I could probably do a better job of optimizing the network as well as exploring other option types

## XGB

XGB performs quite well on initial try. I am going to do some more feature engineering to see if I can increase my score and rmse. (Score referring to kaggle.)
