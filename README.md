# ML-Excercise
Machine learning concept and python library

Titanic-Project:- https://colab.research.google.com/drive/1NB2Scny94fFNAiZqGmGMDNUEckJ_VLH3?usp=sharing

Insurance-Dataset:- https://colab.research.google.com/drive/1-boueOARPJ6Xx9O-6YNJhKYyokE86Thf?usp=sharing

Linear_regression:- https://colab.research.google.com/drive/1aN9qSh6XE_6tnoPxYiWXvQUceCBOo1Z5?usp=sharing

Malaria Dataset:- https://colab.research.google.com/drive/1wW3r4CjoUkKf-vAMwhse5JkHbdtQJMj8?usp=sharing

DCGAN:- https://colab.research.google.com/drive/1wrlCU7JupjmML5jmu70lgzU9exfjXKBq?usp=sharing

Image Segmentation:- https://colab.research.google.com/drive/1JfhfZEx0gesu0O__Z8_DIyUJRzb8hKfN?usp=sharing

Feature Engineering:- https://colab.research.google.com/drive/1fC7aUTXCEuu5j7h5t-pzSnNsfuDrRCIu?usp=sharing

Electronic-Production:- https://colab.research.google.com/drive/1PqsOYzys5VgOG-HkQY6ETD5zn2WJpzel?usp=sharing

Decision Tree topic:- https://colab.research.google.com/drive/1P45wRtpQ5q3QHJ9pk-2l7JFj7SOG8IQd?usp=sharing

Linear Regression using statsmodel:- https://colab.research.google.com/drive/1YiFDEuMFkviBIf3NGNj62W9p8O5YC5uw?usp=sharing

SVM(linear, non-linear):- https://colab.research.google.com/drive/1HVspDPlacA2hDrXdzqP_xILuHVcKM8cV?usp=sharing

Linear_Regression_multiple :- https://colab.research.google.com/drive/1B20W6MqiWnUkVKh26ZTyzLD7-uDIwZ8A?usp=sharing

Gradient Descent in Linear Regression:-
Gradient Descent is an optimization algorithm used in linear regression to find the best-fit line by minimizing prediction error. It works by repeatedly adjusting the model parameters based on the direction of the steepest decrease in the cost function. This iterative approach helps the model converge toward the optimal solution.

Reduces prediction error by minimizing the cost function, typically Mean Squared Error (MSE).
Efficiently finds optimal model parameters, especially for large datasets and high-dimensional data.
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/0ab9616f-e45a-48a7-8332-9a8b6e9bffdc" />

colab link:- https://colab.research.google.com/drive/103YIdJ9SK6lcfRXM1tjRSPiJPvHFWIQ4?usp=sharing


Logistic_regression by Scikit Learn:- https://colab.research.google.com/drive/12cQnzZs7D7Cxr1HvefgVpNs6YeMIrrXg?usp=sharing

## Detecting Multicollinearity with VIF - Python

Multicollinearity occurs when two or more independent variables are highly correlated which leads to unstable coefficient estimates and reduces model reliability. This makes it difficult to identify the individual effect of each predictor on the dependent variable. The Variance Inflation Factor (VIF) is used to detect multicollinearity in regression analysis. In this article, we’ll see VIF and how to use it in Python to identify multicollinearity.
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/00f0f2dc-9ca4-4fbd-b484-218b64bf8651" />

detecting_multicollinearity_using_vif
Detecting Multicollinearity with VIF
Mathematics behind Variance Inflation Factor (VIF)
VIF shows how much the variance of a regression coefficient increases due to multicollinearity. For each variable, we run a regression where that variable becomes the dependent variable and the remaining variables act as predictors. This gives an R-squared(
R 
2
 ) value that tells how well one variable can be predicted using the others.

Formula for VIF is:

VIF= 
1−R 
2
 
1
​
 

R² ranges from 0 to 1.
A higher R² means the variable is highly predictable from other variables -> higher VIF.
If R² is close to 1, the variable is almost fully explained by others -> strong multicollinearity.
Since VIF increases as R² increases, a higher VIF indicates higher multicollinearity. In practice:

VIF > 5 -> noticeable multicollinearity
VIF > 10 -> severe multicollinearity (action needed)
Understanding this formula helps us correctly spot multicollinearity and decide if we should remove or combine variables.

VIF Interpretation
VIF ≈ 1: No correlation with other predictors
1 < VIF ≤ 5: Mild to moderate correlation (usually fine)
VIF > 10: Strong multicollinearity -> take corrective steps
Multicollinearity Detection using VIF in Python
To detect multicollinearity in regression analysis we can implement the Variance Inflation Factor (VIF) using the statsmodels library. This function calculates the VIF value for each feature in the dataset helping us identify multicollinearity.

vif:- https://colab.research.google.com/drive/1mIr577jdmCuwGGKm3JxlNNWyXCXi7dmZ?usp=sharing

Non-Linear SVM - ML:- https://colab.research.google.com/drive/1SaixpItWs8GvUVleLN2HJw_RDD9Bii0B?usp=sharing

Implementing Decision Tree Classifiers with Scikit:-https://colab.research.google.com/drive/1yGblSmc5aarpoeQhCjn9XpYvqKGdMxxJ?usp=sharing

ML - Heart Disease Prediction Using Logistic Regression  comments:- https://colab.research.google.com/drive/1GBpMl-fU0sbVtPiEtNP92ERpHInXcT6A?usp=sharing

## **Gaussian Naive Bayes**

comments
Gaussian Naive Bayes is a type of Naive Bayes method working on continuous attributes and the data features that follows Gaussian distribution throughout the dataset. This “naive” assumption simplifies calculations and makes the model fast and efficient.

It is widely used because it performs well even with small datasets and is easy to implement and interpret.
Works well with continuous numerical features
Fast, simple, and effective for classification problems
Commonly used in spam detection, medical diagnosis, and pattern recognition

GaussianNB:- https://colab.research.google.com/drive/1xuGHJ1Hr-F2sO1YBSzSktffuSd1ke5G6?usp=sharing

## A decision boundary is a line or surface that divides different groups in a classification task. It shows which areas belong to which class based on what the model decides. K-Nearest Neighbours (KNN) algorithm operates on the principle that similar data points exist in close proximity within a feature space.

420046938
Decision Boundaries in KNN
The shape of this boundary depends on:

The value of K (how many neighbours are considered).
How the data points are spread out in space.
For example, given a dataset with two classes the decision boundary can be visualized as the line or curve dividing the two regions where each class is predicted. For a 1-nearest neighbour (1-NN) classifier the decision boundary can be visualized using a Voronoi diagram.

Using Voronoi Diagrams to Visualize
A Voronoi diagram splits space into regions based on which training point is closest.
Each region called a Voronoi cell contains all the points closest to one specific training point.
The lines between regions are where points are equally close to two or more seeds. These are the decision boundaries for 1-Nearest Neighbour which is very irregular in shape.
If we label the training points by class the Voronoi diagram shows how KNN assigns a new point based on which region it falls into.
The boundary line between two points 
p 
i
​
  and 
p 
j
​
 is the perpendicular bisector of the line joining them meaning it’s a line that cuts the segment between them exactly in half at a right angle.
knn-decision-boundafries
Formation of Decision Boundaries
Relationship between KNN and Voronoi
In two-dimensional space the decision boundaries of KNN can be visualized as Voronoi diagrams. Here’s how:

KNN Boundaries: The decision boundary for KNN is determined by regions where the classification changes based on the nearest neighbors. As k increases, the decision boundary becomes smoother and less sensitive to local variations, and for very large k, the model may predict the majority class for most regions.
Voronoi Diagram as a Special Case: When k = 1 KNN’s decision boundaries directly correspond to the Voronoi diagram of the training points. Each region in the Voronoi diagram represents the area where the nearest training point is closest.
How KNN Defines Decision Boundaries
In KNN, decision boundaries are influenced by the choice of k and the distance metric used:

1. Impact of 'K' on Decision Boundaries: The number of neighbors (k) affects the shape and smoothness of the decision boundary.

Small k: When k is small the decision boundary can become very complex, closely following the training data. This can lead to overfitting.
Large k: When k is large the decision boundary smooths out and becomes less sensitive to individual data points, potentially leading to underfitting.
2. Distance Metric: The decision boundary is also affected by the distance metric used like Euclidean, Manhattan. Different metrics can lead to different boundary shapes.

Euclidean Distance: Commonly used leading to circular or elliptical decision boundaries in two-dimensional space.
Manhattan Distance: Results in axis-aligned decision boundaries.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/3e6878f9-7301-4a50-aa93-a6980b7ed776" />
