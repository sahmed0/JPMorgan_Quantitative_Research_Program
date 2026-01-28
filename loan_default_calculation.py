# -*- coding: utf-8 -*-
"""
Loan Default Predictor

1. Feature Scaling: Standardises inputs so 'Income' doesn't overpower 'Debt Ratio'.
2. Stratified Splitting: Ensures the rare 'Default' cases are evenly split.
3. Class Weighting: penalties for missing a default are higher than false alarms.
4. Expected Loss Calculation: Converts probability into financial risk.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt

class LoanDefaultPredictor:
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        self.processed_data = None
        self.pipeline = None
        self.features = ['credit_lines_outstanding', 'debt_to_income', 
                         'payment_to_income', 'years_employed', 'fico_score']
        
    def preprocess(self):
        df = self.raw_data.copy()
        
        # 1. Cleaning: Zeros in income/debt are likely errors, treat as NaN
        cols_to_clean = ['income', 'total_debt_outstanding', 'loan_amt_outstanding']
        for col in cols_to_clean:
            df[col] = df[col].replace(0, np.nan)
            
        # Drop rows where critical financial info is missing
        df = df.dropna(subset=cols_to_clean)
        
        # 2. Feature Engineering
        df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']
        df['debt_to_income'] = df['total_debt_outstanding'] / df['income']
        
        self.processed_data = df
        return df

    def train(self):
        if self.processed_data is None:
            self.preprocess()
            
        X = self.processed_data[self.features]
        y = self.processed_data['default']

        # Split for final validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. Pipeline Construction
        # StandardScaler is CRITICAL for Logistic Regression. 
        # Without it, FICO (range 300-850) dominates Ratios (range 0-1).
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('logistic', LogisticRegression(
                solver='liblinear', 
                class_weight='balanced', # Handles rare defaults automatically
                random_state=42
            ))
        ])

        # 4. Cross-Validation (More robust than single split)
        # Use ROC_AUC because it handles imbalanced classes better than Accuracy
        cv = StratifiedKFold(n_splits=5)
        scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        
        print(f"Cross-Validation AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

        # Fit on full training set
        self.pipeline.fit(X_train, y_train)
        
        # Final Evaluation
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        self._print_evaluation(y_test, y_pred_proba)

    def _print_evaluation(self, y_test, y_probs):
        fpr, tpr, _ = metrics.roc_curve(y_test, y_probs)
        auc_score = metrics.auc(fpr, tpr)
        
        print("\n--- Final Test Set Performance ---")
        print(f"AUC Score: {auc_score:.3f}")
        
        # Plotting (Optional)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def predict_expected_loss(self, loan_amount, recovery_rate=0.1):
        """
        Calculates Expected Loss (EL) = PD * LGD * EAD
        PD: Probability of Default (from model)
        LGD: Loss Given Default (1 - recovery_rate)
        EAD: Exposure at Default (loan_amount)
        """
        if self.pipeline is None:
            raise Exception("Model not trained yet.")
            
        # Create a dummy dataframe for the user input
        # For this example, we predict on the existing test set to show the distribution
        X_data = self.processed_data[self.features]
        probabilities = self.pipeline.predict_proba(X_data)[:, 1]
        
        # Financial metric: Expected Loss
        exposure_at_default = self.processed_data['loan_amt_outstanding']
        loss_given_default = 1 - recovery_rate
        
        expected_losses = probabilities * loss_given_default * exposure_at_default
        
        total_risk = expected_losses.sum()
        print(f"\nTotal Portfolio Expected Loss: ${total_risk:,.2f}")
        return expected_losses

# --- Usage --- #
predictor = LoanDefaultPredictor('Loan_Data.csv')
predictor.train()
losses = predictor.predict_expected_loss(loan_amount=None) # Uses internal data
