import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    # Drop customer ID
    df.drop(columns=['customerID'], inplace=True, errors='ignore')

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # One-hot encode remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)

    print(f"Processed shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. EDA PLOTS
# ─────────────────────────────────────────────

def run_eda(df_raw):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Telecom Churn - EDA Overview', fontsize=16, fontweight='bold')

    # Churn distribution
    df_raw['Churn'].value_counts().plot.pie(
        autopct='%1.1f%%', ax=axes[0, 0],
        labels=['No Churn', 'Churn'], colors=['#4CAF50', '#F44336']
    )
    axes[0, 0].set_title('Churn Distribution')
    axes[0, 0].set_ylabel('')

    # Contract type
    sns.countplot(data=df_raw, x='Contract', hue='Churn', ax=axes[0, 1],
                  palette=['#4CAF50', '#F44336'])
    axes[0, 1].set_title('Churn by Contract Type')
    axes[0, 1].tick_params(axis='x', rotation=15)

    # Internet service
    sns.countplot(data=df_raw, x='InternetService', hue='Churn', ax=axes[0, 2],
                  palette=['#4CAF50', '#F44336'])
    axes[0, 2].set_title('Churn by Internet Service')

    # Monthly charges
    df_raw.groupby('Churn')['MonthlyCharges'].plot(
        kind='hist', alpha=0.6, ax=axes[1, 0],
        legend=True, color=['#4CAF50', '#F44336']
    )
    axes[1, 0].set_title('Monthly Charges Distribution')
    axes[1, 0].set_xlabel('Monthly Charges')

    # Tenure
    df_raw.groupby('Churn')['tenure'].plot(
        kind='hist', alpha=0.6, ax=axes[1, 1],
        legend=True, color=['#4CAF50', '#F44336']
    )
    axes[1, 1].set_title('Tenure Distribution')
    axes[1, 1].set_xlabel('Tenure (Months)')

    # Payment method
    sns.countplot(data=df_raw, x='PaymentMethod', hue='Churn', ax=axes[1, 2],
                  palette=['#4CAF50', '#F44336'])
    axes[1, 2].set_title('Churn by Payment Method')
    axes[1, 2].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig('output/eda_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("EDA plot saved to output/eda_overview.png")


# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────

def train_models(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        'LogisticRegression':        LogisticRegression(),
        'KNeighborsClassifier':      KNeighborsClassifier(),
        'GaussianNB':                GaussianNB(),
        'DecisionTreeClassifier':    DecisionTreeClassifier(),
        'RandomForestClassifier':    RandomForestClassifier(),
        'AdaBoostClassifier':        AdaBoostClassifier(),
        'GradientBoostingClassifier':GradientBoostingClassifier(),
    }

    results = {}
    print("\n── Cross-Validation Results (5-Fold) ──")
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        results[name] = [scores.mean(), scores.std()]
        print(f"{name:30s}  Acc: {scores.mean():.4f} ± {scores.std():.4f}")

    # Voting Classifier
    clf1 = GradientBoostingClassifier()
    clf2 = LogisticRegression()
    clf3 = AdaBoostClassifier()
    voting_clf = VotingClassifier(
        estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)],
        voting='soft'
    )
    scores = cross_val_score(voting_clf, X_scaled, y, cv=5, scoring='accuracy')
    results['VotingClassifier'] = [scores.mean(), scores.std()]
    print(f"{'VotingClassifier':30s}  Acc: {scores.mean():.4f} ± {scores.std():.4f}")

    # Final fit & evaluation
    voting_clf.fit(X_train, y_train)
    predictions = voting_clf.predict(X_test)

    print("\n── Final Model: Voting Classifier ──")
    print(f"Test Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix - Voting Classifier')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved to output/confusion_matrix.png")

    return results, voting_clf


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)

    # Load raw data for EDA
    df_raw = load_data('data/telecom_churn.csv')
    run_eda(df_raw)

    # Preprocess & model
    df_processed = preprocess(df_raw.copy())
    results, final_model = train_models(df_processed)

    print("\n✅ Done! Check the output/ folder for saved plots.")
