# Titanic Survival Prediction
## Project Overview
This project aims to build a machine learning model to predict whether a passenger survived the Titanic disaster. The dataset is taken from the Kaggle Titanic competition and contains demographic and travel-related information about passengers.
The objective of this project is to perform complete exploratory data analysis, handle missing values, engineer meaningful features, train multiple machine learning models, compare their performance, and generate a final submission file suitable for Kaggle evaluation.
This project demonstrates end-to-end machine learning workflow including data preprocessing, feature engineering, model training, evaluation, and prediction generation.

## Dataset
Source: Kaggle Titanic – Machine Learning from Disaster
Files used:
* train.csv (training data with target variable)
* test.csv (test data without target variable)
The training dataset contains 891 records and the test dataset contains 418 records.

## Exploratory Data Analysis
Exploratory data analysis was performed to understand the dataset structure, distributions, and relationships between features and survival.
Key observations:
* Survival rate was higher among female passengers compared to males.
* First-class passengers had a significantly higher survival rate than second and third class.
* Younger passengers showed slightly higher survival rates.
* Family size appeared to influence survival probability.
EDA included:
* Distribution plots
* Survival count analysis
* Survival comparison by gender and passenger class
* Age distribution analysis
* Missing value inspection

## Data Preprocessing
Missing values were handled carefully using appropriate strategies:
* Age: Filled using median age grouped by passenger class.
* Embarked: Filled using mode.
* Fare: Filled using median.
* Cabin: Dropped due to excessive missing values.
Irrelevant columns such as Name, Ticket, and PassengerId were removed after extracting useful information.

## Feature Engineering
Several meaningful features were created to improve model performance:
* FamilySize: Created using SibSp + Parch + 1.
* IsAlone: Indicates whether a passenger traveled alone.
* AgeGroup: Categorized age into meaningful bins.
* Title: Extracted from passenger names to capture social status information.
These engineered features improved model interpretability and predictive performance.

## Model Training
Two models were trained and evaluated:
1. Logistic Regression
2. Random Forest Classifier
A stratified train-test split was used to maintain class balance. Cross-validation was also performed to evaluate model stability.

## Model Evaluation and Comparison
Models were compared using the following metrics:
* Accuracy
* Precision
* Recall
* F1 Score
* Cross-validation score

## Results:
* Logistic Regression achieved 82% accuracy.
* Random Forest achieved 80% accuracy.
Logistic Regression slightly outperformed Random Forest. This suggests that the Titanic dataset contains mostly linear relationships between features and survival. Since the dataset is relatively small, simpler models generalized better.
Random Forest showed slightly higher training accuracy but lower test accuracy, indicating minor overfitting.

## Final Prediction
The best-performing model was trained on the full training dataset and used to generate predictions for the test dataset.
The final submission file was created in the required format:
PassengerId, Survived
The file `submission.csv` is located in the outputs folder.

## Setup Instructions
1. Install dependencies:
pip install -r requirements.txt
2. Run the notebook:
Open notebooks/Titanic_survival_prediction.ipynb and execute all cells.

## Key Learnings
* Importance of thorough exploratory data analysis.
* Impact of feature engineering on model performance.
* Simpler models can outperform complex models on structured datasets.
* Cross-validation is essential for reliable model evaluation.
* Proper project structure improves readability and maintainability.
