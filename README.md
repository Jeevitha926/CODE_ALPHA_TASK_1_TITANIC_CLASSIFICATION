# Titanic Classification: Predicting Survival on the Titanic

This project is part of a data science task focused on building a classification model to predict whether a passenger would survive the Titanic disaster. The dataset includes various attributes like socio-economic status, age, and gender, which may contribute to the survival rate.

## Project Overview

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. In this project, we aim to predict the survival outcome of Titanic passengers using a variety of features like socio-economic status, age, and gender. This project provides insights into factors that influenced survival rates, and serves as a foundational exercise in classification modeling.

## Problem Statement

To predict the survival of a passenger on the Titanic based on various attributes, such as:
- Socio-economic status (e.g., Class)
- Age
- Gender
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Fare paid for the ticket

## Dataset

The dataset is the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data). It contains the following columns:

- PassengerId: ID given to each passenger
- Survived: Survival indicator (1 = Survived, 0 = Did not survive)
- Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Name: Passenger's full name
- Sex: Gender
- Age: Age in years
- SibSp: Number of siblings/spouses aboard
- Parch: Number of parents/children aboard
- Ticket: Ticket number
- Fare: Ticket fare
- Cabin: Cabin number
- Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

## Project Workflow

1. *Data Preprocessing*:
   - Handling missing values
   - Encoding categorical features (e.g., Sex, Embarked)
   - Feature scaling (if necessary)

2. *Exploratory Data Analysis (EDA)*:
   - Analyzing the impact of various features (e.g., Pclass, Sex, Age) on survival
   - Visualizing data to uncover patterns and relationships

3. *Feature Engineering*:
   - Creating new features or modifying existing ones to improve the model's performance
   - Examples: Family size, Title extraction from Name

4. *Modeling*:
   - Using machine learning classification algorithms such as Logistic Regression, Decision Trees, Random Forest, or Support Vector Machines (SVM)
   - Hyperparameter tuning and cross-validation to improve model accuracy

5. *Evaluation*:
   - Evaluating models based on accuracy, precision, recall, and F1-score
   - Choosing the best model based on performance metrics

## Results and Findings

The analysis revealed that certain factors were strong predictors of survival:
- *Gender*: Females had a higher survival rate than males.
- *Passenger Class*: Higher-class passengers had a higher survival rate.
- *Age*: Younger passengers had a slightly higher chance of survival.

## Installation

To run this project locally, ensure you have Python installed along with the necessary libraries:

bash
pip install pandas numpy matplotlib seaborn scikit-learn


## Usage

1. Clone the repository:
   bash
   git clone https://github.com/Jeevitha926/titanic-classification.git
   
2. Navigate to the project directory and open the Jupyter Notebook:
   bash
   cd titanic-classification
   jupyter notebook titanic_classification.ipynb
   
3. Follow the steps in the notebook to preprocess data, explore features, train models, and evaluate results.
