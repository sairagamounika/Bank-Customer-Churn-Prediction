# Bank-Customer-Churn-Prediction

## Problem Statement
Customer churn is a critical challenge faced by businesses, particularly in the banking sector, where retaining customers directly impacts profitability. This project aims to predict whether a customer will leave the bank (churned: Exited=1) or stay (not churned: Exited=0), based on various customer features, including demographics, account details, and service usage patterns. By accurately predicting customer churn, banks can take proactive measures to improve customer retention and minimize losses.
## Data Setup
The dataset used is the Bank Customer Churn Prediction Dataset from Kaggle. It contains customer records, each with features such as customer demographics, credit score, account balance, and churn status. The following preprocessing steps were undertaken:
#### 1.	Dummy Variables:
- Irrelevant columns such as RowNumber, CustomerId, and Surname were removed.
- Categorical variables (e.g., Geography, Gender) were encoded using one-hot encoding to prepare them for machine learning models.
#### 2.	Target and Feature Split:
- Target: The "Exited" column, which indicates whether the customer churned or not (binary classification).
- Features: The remaining columns after preprocessing.
#### 3.	Data Splitting:
- Training Set: 70% of the data.
- Test Set: 30% of the data, ensuring class stratification to maintain the balance between churned and non-churned customers.
#### 4.	Scaling:
- The features were normalized using MinMaxScaler to scale the data between 0 and 1, ensuring fair comparisons and preventing any one feature from dominating the model due to its scale.
## Exploratory Data Analysis (EDA)
#### 1.	Churn Distribution:
- A count plot was used to visualize the class imbalance in the target variable.
- Approximately 20% of customers had churned (Exited=1), while 80% remained (Exited=0).
#### 2.	Correlation Analysis:
-	A heatmap was plotted to examine the correlation between features and the target variable.
-	Features such as Geography_Germany, Age, and Balance showed significant correlations with churn.
#### 3.	Feature Distributions:
- Histograms for numerical features revealed skewness in features such as Balance and EstimatedSalary, which might require further transformations for better model fitting.
## Model Performance Evaluation
Several machine learning models were trained to predict customer churn, and their performance was evaluated using various metrics, including accuracy, precision, recall, and F1-score.
### 1. Logistic Regression
The Logistic Regression model initially achieved an accuracy of 81.4% on the test set. However, the performance for the minority class (churned customers) was limited, with a precision of 65% and a recall of 19%, indicating poor sensitivity to the minority class. After applying resampling techniques to address class imbalance, the recall for churned customers increased significantly to 73%, though overall accuracy decreased to 71.7%.
### 2. Decision Tree
The Decision Tree model performed with an accuracy of 86.1% on the test set. It showed better handling of the minority class compared to Logistic Regression, with a precision of 72% and a recall of 52%. After resampling, the accuracy reduced to 77%, but the minority class recall increased to 72%, highlighting an improved balance between the two classes. Hyperparameter tuning (max depth: 6, min samples leaf: 5, min samples split: 2) further optimized the Decision Tree, achieving an accuracy of 86.2% with balanced performance across both classes.
### 3. Random Forest
The Random Forest model demonstrated robust performance with a test set accuracy of 87.1%. It achieved a precision of 82% and a recall of 47% for the minority class. Post-resampling, the recall for the minority class improved significantly to 79%, with a slight drop in accuracy (79.4%).
### 4. Gradient Boosting
Gradient Boosting achieved the highest accuracy of 87.3% on the test set, with a precision of 80% and a recall of 50% for the minority class. After resampling, the accuracy was 80%, and the minority class recall improved to 79%, further affirming the model's adaptability to imbalanced datasets. The Gradient Boosting model also exhibited strong performance during cross-validation, with a mean accuracy of 86.2% ± 0.6%.
### 5. Voting Classifier Ensemble
The Voting Classifier ensemble, combining Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting,performed well with an accuracy of 86.7%. However, its recall for the minority class was 43%, and after resampling, recall improved to 79%.
## Key Observations
- **Impact of Class Imbalance:** The dataset’s class imbalance (80% retained vs. 20% churned) posed challenges for models in predicting churned customers accurately. The random undersampling technique was essential to improve the model’s recall for churned customers.
- **Feature Importance:** Features like Age, Balance, Salary, and CreditScore were identified as the most important features for churn prediction, which aligns with business intuition—customers with lower balances or less favorable demographics are more likely to churn.
- **Model Comparison:** Gradient Boosting and Random Forest performed the best in terms of both accuracy and recall for the minority class. These ensemble methods handled class imbalance effectively and offered the best trade-off between precision and recall.
## Areas for Improvement
**1.	Feature Engineering:** Exploring interaction terms between features, such as Age * Balance could capture hidden relationships and improve model accuracy.
#### **2.	Evaluation Metrics:**
- Incorporating metrics like ROC-AUC for imbalanced data might provide a better performance evaluation.
- Trying Support Vector Machines could provide additional insights into the best model for this use case.
## Conclusion
The Bank Customer Churn Prediction project successfully built predictive models to determine the likelihood of a customer churning based on various demographic and account-related features. Through the application of Random Forest, Gradient Boosting, and other ensemble techniques, the project identified the key factors influencing customer churn and demonstrated the importance of handling class imbalance through resampling techniques. Random Forest and Gradient Boosting performed the best in predicting churn, providing a good balance between accuracy and recall. The insights gained from this project can assist banks in targeting high-risk customers and implementing retention strategies to minimize customer churn.
