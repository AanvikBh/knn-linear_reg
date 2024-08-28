## SMAI Assignment 1 Report 
### Aanvik Bhatnagar
### 2022101103

Note: The file 'a1.py' is just the compiled form of all the codes so the paths might not be correct. Do not run it now as it will run for many hours. 

## Task 1: Data Preprocessing and Plotting 
- Scatter plots were generated for every pair of feature to study deep relation between 2 specific features
- Histogram for both numeric and categorical data drawn to understand the distribution
- Liveness , loudness and speechiness have very similar or close values for all the targets
- No specific outliers or skewed data points were found 
- Though there were 1 track genre (indexed 18 ) which showed highly large value for the speechiness
- Few genres (2-3) also had skewed values for loudness and liveness, though all the other lied in near to each other
- A correlation matrix heat map was also made to view how changing a particular feature, affects other features or the target values
- It showed that danceability, key, mode and time_signature didnt have much affect on the target variable 
- A heirarchy structure can also be built using the values of the `track_genre` column in the matrix
- Speechiness > acousticness > instrumentalness > valence > explicit in order of importance for target prediction being the top few independent variables
- Normalization of data also done to see which features should be normalized. Plotted with the original data as a scatter plot. 
- Categorical Data not so helpful for calculating distance, so it can be dropped. Also goes for Unnamed. 
- Data specific info in dataset.ipynb 

## Task 2: KNN
### 2.3.1
k value = 19
Distance Metric used = Manhattan (Vectorized)
Inference Time: 92.93741011619568 s
Performance Metrics obtained:
{'accuracy': 0.19757075997325607, 'precision_macro': 0.1894691274807252, 'recall_macro': 0.1850577053980094, 'f1_score_macro': 0.1872374361315716, 'precision_micro': 0.19757075997325607, 'recall_micro': 0.19757075997325607, 'f1_score_micro': 0.19757075997325604}

k value = 19
Distance Metric used = Manhattan (Normal)
Inference Time: 2160.494178056717 s
Performance Metrics obtained:
{'accuracy': 0.19757075997325607, 'precision_macro': 0.1894691274807252, 'recall_macro': 0.1850577053980094, 'f1_score_macro': 0.1872374361315716, 'precision_micro': 0.19757075997325607, 'recall_micro': 0.19757075997325607, 'f1_score_micro': 0.19757075997325604}


### 2.4.1
{k,distance} = range(10,41) x {'euclidean', 'manhattan', 'cosine'}

Best Pair: {12, manhattan}

Top 10 {k, distance metric} pairs based on validation accuracy:
Rank 1: k=12, metric=manhattan, accuracy=0.2006, processing time: 91.5786 seconds
Rank 2: k=11, metric=manhattan, accuracy=0.2004, processing time: 89.1226 seconds
Rank 3: k=16, metric=manhattan, accuracy=0.2001, processing time: 91.0607 seconds
Rank 4: k=17, metric=manhattan, accuracy=0.1995, processing time: 91.7099 seconds
Rank 5: k=13, metric=manhattan, accuracy=0.1994, processing time: 171.3577 seconds
Rank 6: k=18, metric=manhattan, accuracy=0.1988, processing time: 91.1969 seconds
Rank 7: k=15, metric=manhattan, accuracy=0.1981, processing time: 91.7165 seconds
Rank 8: k=19, metric=manhattan, accuracy=0.1976, processing time: 91.7499 seconds
Rank 9: k=23, metric=manhattan, accuracy=0.1972, processing time: 90.6404 seconds
Rank 10: k=14, metric=manhattan, accuracy=0.1970, processing time: 91.2799 seconds

k vs accuracy graph: Given in 1/figures/knn

Columns which can be removed: time_sig and duration_ms 
New Accuracy: 
{'accuracy': 0.20079088477824829, 'precision_macro': 0.19285527089614388, 'recall_macro': 0.1921779948374726, 'f1_score_macro': 0.19251603720027696, 'precision_micro': 0.20069088477824829, 'recall_micro': 0.20069088477824829, 'f1_score_micro': 0.20069088477824829}



### 2.5.1
Common Model used: {k=19, metric=manhattan}

Initial KNN: Distance Metrics were calculated using a for loop for every row, and then converted to array. 
Vectorized KNN: Using Numpy operations on the matrix X_train and test_row

Time taken by each KNN Model:
Initial Inference Time (s): 2160.494178056717
Optimised Inference Time (s): 92.93741011619568
Sklearn Inference Time (s): 4.132826805114746

Graph for inference time: 1/figures/knn

Train Data Size: 1/figures/knn

### 2.6
Model used: {12, manhattan}

Performance Metrics on Validation Data:
{'accuracy': 0.05368421052631579, 'precision_macro': 0.05144958273913892, 'recall_macro': 0.05461816184395657, 'f1_score_macro': 0.052986544550286965, 'precision_micro': 0.05368421052631579, 'recall_micro': 0.05368421052631579, 'f1_score_micro': 0.05368421052631579}


## Task 3: Linear Regression 

### 3.1.1 
Train Test Val split done in data/interim/1/linear-reg in csv files after shuffling in the beginning so that comparison is done uniformly across all the models. 

Variance of Training Data: 0.45778604930834293	 Standard Dev of Training Data: 0.676598883614467
Variance of Test Data: 0.5831075338722748	 Standard Dev of Test Data: 0.7636147810724166
Variance of Validation Data: 0.44790890132599676	 Standard Dev of Validation Data: 0.6692599654289779

Scatter Plot: 1/figures/linear_reg_plot

Linear Regression with Degree 1: 1/figures/linear_reg_plot

- Plots for Degree>1 have not been saved, for  1<=Degree<=5 animations are saved.
- Number of Iterations of LR kept to 150 across all the models. 
- The moment MSE difference in iterations is less than 0.0001 then LR is stopped. (not done when creating animations to show MSE increase with higher iterations)
- The degree with the minimum MSE is 13.0 with an MSE of 0.0102
- MSE vs Degree Plot: 1/figures/linear_reg_plot

- Values of Variance and Std Dev on Val Data: 1/figures/linear_reg_plot


### Regularisation
- Regularization reduces the variance of the model. In high-dimensional spaces, without regularization, the model might have high variance, leading to overfitting. By shrinking the coefficients, regularization makes the model less sensitive to the specific training data, reducing variance.
- Plots of L1 and L2 regularised polynomials from 5 to 20 degree are in figures/linear-reg directory 
- Values of variance before and after regularisation: 1/figures/linear_reg_plot
- Preferred regularisation: L2 


