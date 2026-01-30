# J.P. Morgan Quant Research Program

This repository contains my solutions for the quantitative research tasks focused on predictive modelling and risk assessment. The projects demonstrate the application of machine learning and statistical optimisation to solve pricing and credit risk problems.

## Projects

### 1. Natural Gas Price Prediction & Contract Valuation

A time-series forecasting model designed to estimate future natural gas prices and value storage contracts.

* **Objective:** Predict prices using historical data and optimise storage injection/withdrawal strategies to maximise profit.
* **Methodology:**
* **Simultaneous Regression:** Uses a custom linear model that fits long-term trends and annual seasonal cycles (Sine/Cosine waves) concurrently.
* **Contract Logic:** A valuation engine that respects physical constraints (injection/withdrawal rates, max storage) and daily storage costs.


* **Key Skills:** Time-series analysis, Seasonal decomposition, Financial modelling, Python (Pandas, NumPy).

### 2. Loan Default Estimation & Risk Segmentation

A dual-module project for estimating credit risk and optimising borrower segmentation.

* **Module A (Default Probability):** A Logistic Regression pipeline to predict the likelihood of loan default.
* *Features:* Includes **Expected Loss (EL)** calculation () and Stratified K-Fold validation.
* *Debugging:* Contains specific logic to identify and handle "toy data" leakage (e.g., perfect separation by synthetic features).


* **Module B (FICO Segmentation):** A statistical script using **Maximum Likelihood Estimation (MLE)**.
* *Objective:* Determine the optimal FICO score cut-offs to bucket borrowers into distinct risk categories.
* *Optimisation:* Uses the **Powell** method (derivative-free) and vectorised cumulative sums for high-performance processing.


* **Key Skills:** Logistic Regression, MLE, Feature Engineering, ROC/AUC analysis, Scipy (Optimisation), Scikit-learn.

## Tools & Libraries Used

* **Core:** Python 3.8+
* **Data Manipulation:** `pandas`, `numpy`
* **Modelling:** `scikit-learn` (Logistic & Linear Regression), `scipy` (Optimise)
* **Visualisation:** `matplotlib`, `seaborn`

## Installation

To run these projects, ensure the required packages are installed:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn

```

## 📄 License
Copyright © 2026 Sajid Ahmed. All Rights Reserved.

This repository is intended solely for portfolio review and recruitment purposes. By accessing this repository, you acknowledge the following terms:
- View Only: Permission is granted to view the source code for the purpose of evaluating my professional skills and experience.
- No Unauthorised Use: No permission is granted to copy, modify, distribute, or use this code for any personal, commercial, or educational project.
- No AI Training: Use of this source code for the purpose of training machine learning models or generative AI is strictly prohibited.
