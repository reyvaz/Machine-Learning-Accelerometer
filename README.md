
## Introduction

This repository contains the R code for different machine learning algorithms. Accelerometer data is used to predict how well individuals perform weight-lifting exercises. The dataset comes from Veloso et al., (2013) and it contains data from accelerometers on the belt, forearm, arm, and dumbbell from 6 individuals.  

The dataset comprises information on 6 participants who were asked to perform one set of 10 repetitions of the unilateral dumbbell biceps curl in five different ways: according to specification (class A), throwing the elbows to the front (B), lifting the dumbbell only halfway (C), lowering the dumbbell only halfway (D), and throwing the hips to the front (E). More information on the dataset can be found at [LINK](http://groupware.les.inf.puc-rio.br/har).   

The code processes data and constructs 3 different machine learning algorithms, including CART, Random Forest, and Boosted GBM to predict how well the dumbbell biceps curl were performed (variable classe in the dataset).   

The training dataset is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). And the test dataset is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).    

* The rendered html version of the code is found at [barbellRmd.html](https://reyvaz.github.io/MachineLearningWL/barbellRmd.html). 

* The full code is at the [barbellRmd.Rmd](https://reyvaz.github.io/MachineLearningWL/barbellRmd.Rmd) file.
