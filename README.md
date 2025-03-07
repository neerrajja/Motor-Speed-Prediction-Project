### Electric Motor Speed Prediction

Overview

This project aims to develop a machine learning model to predict the speed of an electric motor based on sensor data. The dataset used contains real-world measurements from a Permanent Magnet Synchronous Motor (PMSM), including features such as voltage, current, and temperature.

### Dataset

The dataset consists of sensor readings collected from PMSM motors and includes the following features:

Voltage: Input voltage applied to the motor

Current: Electric current passing through the motor

Temperature: Thermal readings from the motor components

Torque: Mechanical torque applied to the motor

Speed: Rotational speed of the motor (target variable)

### Objective

The objective of this project is to train a predictive model that accurately estimates motor speed given various input features.

### Technologies Used

Python: Programming language

Pandas: Data manipulation

NumPy: Numerical computations

Matplotlib/Seaborn: Data visualization

Scikit-learn: Machine learning algorithms

TensorFlow/PyTorch (if deep learning models are used)

### Model Building

The project explores multiple machine learning models, including:

Linear Regression

Lasso Regression and Ridge Regression

Random Forest Regressor

Gradient Boosting Machines (LightGBM)

KNN

Multi Layer Perceptron (MLP)

Decision Tree

### Steps Involved:

Data Preprocessing

Handling missing values

Feature engineering

Data normalization and scaling

Exploratory Data Analysis (EDA)

Understanding feature distributions

Identifying correlations

Detecting outliers

Model Training & Evaluation

Splitting dataset into training and testing sets

Training machine learning models

Evaluating performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)

#### Results

The best-performing model is selected based on evaluation metrics, and predictions are compared against actual speed values to measure accuracy.
