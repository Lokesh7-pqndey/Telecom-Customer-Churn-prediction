<div align="right">
[1]: https://github.com/Lokesh7-pqndey
[2]: https://www.linkedin.com/in/lokesh-pandey-2265b65218
[3]: https://public.tableau.com/app/profile/Lokesh7-pqndey
[4]: https://twitter.com/Lokesh7-pqndey
[![github](https://raw.githubusercontent.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/main/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/main/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/main/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/main/icons/iconmonstr-twitter-5.svg)][4]
</div>

# <div align="center">Telecom Customer Churn Prediction</div>

![Banner](output/customer%20churn.jpeg)

---
## Why Does Churn Matter?
In today's telecom landscape, customers have more choices than ever. They can effortlessly compare plans and switch providers with minimal friction. As a result, telecom companies face an annual churn rate of **15–25%**, making retention one of the highest-priority business challenges in the industry.

Mass retention campaigns are expensive and inefficient. A smarter approach is to identify *which specific customers* are likely to leave — and intervene early. This project builds a machine learning pipeline to do exactly that: flag high-risk customers before they walk out the door.

Retaining an existing customer costs significantly less than acquiring a new one. Every customer retained directly improves profitability, lowers initiation costs, and grows the network's overall value.

---
## Project Goals
- Quantify the proportion of churned vs. retained customers in the dataset
- Identify the strongest behavioural and demographic indicators of churn
- Train and compare multiple classification models using cross-validation
- Deploy the best-performing ensemble model as the final predictor

---
## Dataset
**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/bhartiprasad17/customer-churn-prediction/data)

The dataset covers a snapshot of customer activity across the following dimensions:
- **Churn status** — whether the customer left within the last month
- **Subscribed services** — phone lines, internet type, streaming, security, backup, device protection, tech support
- **Account details** — tenure, contract length, payment method, billing preferences, monthly and total charges
- **Demographics** — gender, age group, partner and dependent status

---
## Tech Stack
**Libraries:** `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---
## Exploratory Data Analysis

### 1. Overall Churn Rate
![Churn Distribution](output/Churn%20Distribution.png)

### 2. Churn by Gender
![Gender Distribution](output/distributionWRTGender.PNG)

### 3. Contract Type
![Contract Distribution](output/Contract%20distribution.png)

### 4. Payment Methods
![Payment Methods](output/payment%20methods.png)
![Payment Methods vs Churn](output/payment%20methods%20with%20respect%20to%20churn.png)

### 5. Internet Service Type
![Internet Services](output/internet%20services.PNG)

### 6. Dependents
![Dependents](output/dependents.PNG)

### 7. Online Security
![Online Security](output/onlineSecurity.PNG)

### 8. Senior Citizens
![Senior Citizens](output/seniorCitizen.PNG)

### 9. Paperless Billing
![Billing](output/billing.PNG)

### 10. Tech Support
![Tech Support](output/techSupport.PNG)

### 11. Monthly Charges, Total Charges & Tenure
![Monthly Charges](output/carges%20distribution.PNG)
![Total Charges](output/total%20charges.PNG)
![Tenure](output/tenure%20and%20churn.PNG)

### 12. Correlation with Churn
![Correlation](output/correlation%20with%20churn.PNG)

### 13. ROC AUC Comparison
![ROC AUC](output/ROC%20AUC%20comparison.PNG)

### 14. Accuracy Score Comparison
![Accuracy Comparison](output/Accuracy%20score%20comparison.PNG)

---
## Machine Learning Models
![Model Evaluation](output/Model%20evaluation.PNG)

### Results after K-Fold Cross Validation
![Logistic Regression](output/LR.PNG)
![KNN](output/KNN.PNG)
![Naive Bayes](output/Naive%20Bayes.PNG)
![Decision Tree](output/Decision%20trees.PNG)
![Random Forest](output/Random%20Forest.PNG)
![Adaboost](output/Adaboost.PNG)
![Gradient Boost](output/Gradient%20boost.PNG)
![Voting Classifier](output/Voting%20Classifier.PNG)

![All Confusion Matrices](output/confusion_matrix_models.PNG)

---
## Final Model: Voting Classifier
The three strongest individual classifiers — Gradient Boosting, Logistic Regression, and AdaBoost — were combined into a **soft-voting ensemble** for the final prediction model.

```python
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()

final_model = VotingClassifier(
    estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)],
    voting='soft'
)

final_model.fit(X_train, y_train)
predictions = final_model.predict(X_test)
print("Final Accuracy:", accuracy_score(y_test, predictions))
