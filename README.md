# Telecom Customer Churn Prediction

[![GitHub](https://img.shields.io/badge/GitHub-Lokesh7--pqndey-black?logo=github)](https://github.com/Lokesh7-pqndey)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Lokesh%20Pandey-blue?logo=linkedin)](https://www.linkedin.com/in/lokesh-pandey-2265b5218)
[![Tableau](https://img.shields.io/badge/Tableau-Portfolio-orange?logo=tableau)](https://public.tableau.com/app/profile/Lokesh7-pqndey)

![Banner](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/customer%20churn.jpeg?raw=true)

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

![Churn Distribution](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Churn%20Distribution.png?raw=true)

> Nearly **1 in 4 customers** (26.6%) churned — a significant signal that proactive retention strategies are urgently needed.

---

### 2. Churn by Gender

![Gender Distribution](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/distributionWRTGender.PNG?raw=true)

> Male and female customers churned at virtually identical rates. Gender alone is not a reliable churn predictor.

---

### 3. Contract Type

![Contract Distribution](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Contract%20distribution.png?raw=true)

> Customers on **Month-to-Month** contracts accounted for ~75% of all churners. Those on longer-term contracts showed dramatically lower attrition — 13% for one-year and just 3% for two-year plans.

---

### 4. Payment Methods

![Payment Methods](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/payment%20methods.png?raw=true)

![Payment Methods vs Churn](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/payment%20ethods%20with%20respectto%20churn.PNG?raw=true)

> **Electronic Check** users had the highest churn rate. Customers on automatic payment methods (bank transfer or credit card) were considerably more loyal.

---

### 5. Internet Service Type

![Internet Services](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/internet%20services.PNG?raw=true)

> **Fiber Optic** subscribers showed a disproportionately high churn rate vs. DSL users — possibly pointing to unmet expectations around speed, reliability, or value.

---

### 6. Dependents

![Dependents](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/dependents.PNG?raw=true)

> Customers without dependents are far more likely to switch providers, possibly due to lower switching costs.

---

### 7. Online Security

![Online Security](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/onlineSecurity.PNG?raw=true)

> The absence of an online security subscription is strongly linked with churn. Security features appear to act as a **retention anchor**.

---

### 8. Senior Citizens

![Senior Citizens](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/seniorCitzen.PNG?raw=true)

> While senior citizens make up a small share of the total base, they churn at a notably higher rate.

---

### 9. Paperless Billing

![Billing](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/billing.PNG?raw=true)

> Paperless billing customers show higher churn. These tend to be digital-native users who are more likely to shop around online.

---

### 10. Tech Support

![Tech Support](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/techSupport.PNG?raw=true)

> Customers with no tech support subscription are among the most at-risk. Unresolved technical frustrations are a key churn trigger.

---

### 11. Monthly Charges, Total Charges & Tenure

![Monthly Charges](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/carges%20distribution.PNG?raw=true)

![Total Charges](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/total%20charges.PNG?raw=true)

![Tenure](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/tenure%20and%20churn.PNG?raw=true)

> Higher monthly charges correlate strongly with churn risk. Brand new customers (low tenure) are also significantly more likely to leave.

---

### 12. Correlation with Churn

![Correlation](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/correlation%20with%20churn.PNG?raw=true)

---

### 13. ROC AUC Comparison

![ROC AUC](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/ROC%20AUC%20comparison.PNG?raw=true)

---

### 14. Accuracy Score Comparison

![Accuracy Comparison](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Accuracy%20score%20comparison.PNG?raw=true)

---

## Machine Learning Models

![Model Evaluation](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Model%20evaluation.PNG?raw=true)

### Results after K-Fold Cross Validation

![Logistic Regression](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/LR.PNG?raw=true)
![KNN](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/KNN.PNG?raw=true)
![Naive Bayes](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Naive%20Bayes.PNG?raw=true)
![Decision Tree](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Decision%20trees.PNG?raw=true)
![Random Forest](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Random%20Forest.PNG?raw=true)
![Adaboost](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Adaboost.PNG?raw=true)
![Gradient Boost](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Gradient%20boost.PNG?raw=true)
![Voting Classifier](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/Voting%20Classifier.PNG?raw=true)
![All Confusion Matrices](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/confusion_matrix_models.PNG?raw=true)

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
```

### Model Accuracy Results

| Model | Accuracy | Std Dev |
|-------|----------|---------|
| Logistic Regression | 0.8413 | ±0.0105 |
| KNeighborsClassifier | 0.7913 | ±0.0082 |
| GaussianNB | 0.8232 | ±0.0074 |
| DecisionTreeClassifier | 0.6470 | ±0.0220 |
| RandomForestClassifier | 0.8198 | ±0.0116 |
| AdaBoostClassifier | 0.8446 | ±0.0113 |
| GradientBoostingClassifier | 0.8446 | ±0.0107 |
| **VotingClassifier** | **0.8468** | **±0.0109** ✅ |

### Final Confusion Matrix

![Confusion Matrix](https://github.com/Lokesh7-pqndey/Telecom-Customer-Churn-prediction/blob/main/output/confusion%20matrix.PNG?raw=true)

> Out of **1,549 actual non-churn** customers, the model correctly identified 1,400. Out of **561 actual churn** customers, it correctly flagged 281.

---

## Potential Improvements

- **Hyperparameter Tuning** — Apply GridSearchCV or Optuna to the ensemble
- **Feature Engineering** — Create interaction terms such as charge-per-tenure-month ratios
- **Class Imbalance Handling** — Use SMOTE or class weighting to boost churn recall
- **Model Explainability** — Integrate SHAP values for business-ready interpretability

---

## Repository Structure

```
Telecom-Customer-Churn-prediction/
│
├── icons/                          ← SVG social media icons
├── output/                         ← All generated plots and figures
├── Customer churn prediction.ipynb ← Jupyter notebook with full analysis
├── model.py                        ← Python ML pipeline script
├── data.csv                        ← Raw dataset
├── requirements.txt                ← Python dependencies
└── README.md
```

---

## Feedback

Have suggestions or found something to improve? Open an issue or reach out at **pandeylokesh87@gmail.com**

---

## About Me

### Hi, I'm Lokesh Pandey! 👋

I'm a Data Science & ML enthusiast passionate about turning raw data into actionable business insights.

[![GitHub](https://img.shields.io/badge/GitHub-Lokesh7--pqndey-black?logo=github)](https://github.com/Lokesh7-pqndey)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Lokesh%20Pandey-blue?logo=linkedin)](https://www.linkedin.com/in/lokesh-pandey-2265b5218)
[![Tableau](https://img.shields.io/badge/Tableau-Portfolio-orange?logo=tableau)](https://public.tableau.com/app/profile/Lokesh7-pqndey)
