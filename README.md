# Classifying Dangerous Asteroids using Machine Learning

This project aims to classify asteroids as hazardous or non-hazardous based on their physical and orbital characteristics. The dataset used in this project was obtained from NASA and can be found on [Kaggle](https://www.kaggle.com/datasets/basu369victor/prediction-of-asteroid-diameter).

## Getting Started

To run this project, you will need to have the following Python packages installed:

> Scikit-learn (sklearn)
> Seaborn
> Matplotlib
> NumPy

You can install these packages using pip:

`pip install sklearn seaborn matplotlib numpy`

After installing the required packages, you can run the notebooks to train and evaluate the models.

> src/knn.ipynb
> src/lr.ipynb
> src/svc.ipynb

## Dataset

The dataset used in this project contains information about asteroids that have been observed by NASA.

The dataset includes a binary target variable `pha -> potentially hazardous asteroid` indicating whether each asteroid is classified as hazardous or non-hazardous.

## Models

In this project, three classification models were trained and evaluated using the dataset:

> Logistic Regression
> Support Vector Machines (SVM)
> K-Nearest Neighbors (KNN)

The performance of each model was evaluated using accuracy, precision, recall, and F1-score metrics. The best-performing model was chosen based on these metrics.

## Results

The best-performing model was the Linear Regression classifier, with an accuracy of 0.99, precision of 0.94, recall of 1.00, and F1-score of 0.97. The Linear Regression model was able to accurately classify hazardous and non-hazardous asteroids based on their physical and orbital characteristics.

## Conclusion

This project demonstrates the potential for Machine Learning to be used in the field of astronomy to identify and classify potentially hazardous objects in space.