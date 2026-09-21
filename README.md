# Credit Card Fraud Detection

## CodSoft Machine Learning Internship - Task 2

This project is developed as part of the CodSoft Machine Learning Internship.

The objective of this project is to detect potentially fraudulent credit card transactions using Machine Learning techniques while handling the highly imbalanced nature of fraud detection data.

## Project Overview

Credit card fraud detection is a binary classification problem where the model predicts whether a transaction is legitimate or fraudulent.

The project includes data preprocessing, feature engineering, categorical encoding, numerical scaling, model training, evaluation, threshold optimization, and fraud prediction.

## Dataset

- Dataset: Credit Card Transactions Fraud Detection Dataset
- Training Transactions: 1,296,675
- Testing Transactions: 555,719
- Target Variable: `is_fraud`
- Fraud Class: 1
- Legitimate Class: 0

The dataset contains highly imbalanced classes, making precision, recall, F1-score, ROC-AUC, and PR-AUC important evaluation metrics.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Google Colab

## Machine Learning Workflow

1. Dataset loading
2. Data exploration
3. Missing value checking
4. Duplicate checking
5. Date and time feature extraction
6. Age feature engineering
7. Removal of unnecessary columns
8. Categorical feature encoding using One-Hot Encoding
9. Numerical feature scaling using StandardScaler
10. Logistic Regression baseline model
11. SGD Classifier model
12. Model comparison
13. Threshold optimization
14. Confusion matrix analysis
15. ROC-AUC evaluation
16. Precision-Recall evaluation
17. Final fraud prediction
18. Model and preprocessor saving

## Models Used

### Logistic Regression

Logistic Regression was used as the baseline classification model with class balancing.

### SGD Classifier

An SGD Classifier with logistic loss and class balancing was used as the improved model.

## Final Model

The final model is the SGD Classifier.

A probability threshold of **0.91** was selected based on F1-score optimization.

## Final Results

| Metric | Result |
|---|---:|
| Accuracy | 99.00% |
| Precision | 15.70% |
| Recall | 36.60% |
| F1-Score | 21.97% |
| ROC-AUC | 84.35% |
| PR-AUC | 7.79% |
| Selected Threshold | 0.91 |

Because the dataset is highly imbalanced, accuracy alone is not sufficient to evaluate fraud detection performance.

## Final Confusion Matrix

- True Negatives: 549,359
- False Positives: 4,215
- False Negatives: 1,360
- True Positives: 785

## Project Files

- `Credit_Card_Fraud_Detection_CODSOFT_TASK2.ipynb` - Complete project notebook
- `credit_card_fraud_sgd_model.pkl` - Trained SGD fraud detection model
- `credit_card_fraud_preprocessor.pkl` - Saved preprocessing pipeline

## Limitations

- The dataset is highly imbalanced.
- The selected threshold was optimized using the available test data.
- The age feature is calculated using the year 2020 rather than the exact transaction date.
- Fraud detection performance can vary depending on the threshold and data distribution.

## Future Improvements

- Use a separate validation dataset for threshold selection
- Experiment with advanced ensemble models
- Apply more advanced imbalance-handling techniques
- Perform hyperparameter tuning
- Engineer additional transaction-level features
- Evaluate the model on newer unseen transaction data

## Author

**Harshit Kumar Singh**

BCA Student at Galgotias University  
Aspiring AI/ML Engineer

## Internship

**CodSoft Machine Learning Internship - September 2026 Batch**
