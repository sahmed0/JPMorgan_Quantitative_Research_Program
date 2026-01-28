# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 23:45:49 2023

@author: ahmed

Optimises FICO score segmentation to maximise the log-likelihood of default
prediction.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

class DefaultLikelihoodOptimizer:
    def __init__(self, file_path):
        # Load the data and prepare the cumulative arrays immediately.
        # This prevents recalculating the sums during every iteration of the optimiser.
        self.df = pd.read_csv(file_path)
        self._prepare_cumulative_data()

    def _prepare_cumulative_data(self):
        """
        Creates a 'Summed Area Table' (Cumulative Sum) for O(1) lookups.
        This allows us to instantly calculate defaults in the range [a, b].
        """
        # Group by FICO score to get counts per score
        # FICO scores range from 300 to 850.
        grouped = self.df.groupby('fico_score')['default'].agg(['sum', 'count'])
        
        # Reindex to ensure every possible FICO score (300-850) is represented, filling 0s
        all_scores = pd.Index(range(300, 851), name='fico_score')
        grouped = grouped.reindex(all_scores, fill_value=0)
        
        # Calculate cumulative sums
        self.cum_defaults = grouped['sum'].cumsum().values
        self.cum_total = grouped['count'].cumsum().values
        
        # Store boundaries
        self.min_fico = 300
        self.max_fico = 850

    def _get_bin_stats(self, start_idx, end_idx):
        """
        Helper to get total defaults (k) and total loans (n) between two scores.
        """
        # Adjust indices to match the 0-indexed array (300 maps to index 0)
        s = int(start_idx) - self.min_fico
        e = int(end_idx) - self.min_fico
        
        # Handle boundary conditions
        s = max(0, s)
        e = min(len(self.cum_total) - 1, e)
        
        if s > 0:
            k = self.cum_defaults[e] - self.cum_defaults[s - 1]
            n = self.cum_total[e] - self.cum_total[s - 1]
        else:
            k = self.cum_defaults[e]
            n = self.cum_total[e]
            
        return k, n

    def negative_log_likelihood(self, points):
        """
        Computes the negative log-likelihood for a given set of boundaries.
        """
        # 1. Clean and sort the points to ensure valid segmentation
        points = np.sort(points)
        
        # 2. Define the full set of boundaries: Start -> Points -> End
        # We implicitly start at 300 and end at 850
        boundaries = [self.min_fico] + list(points) + [self.max_fico]
        
        log_likelihood_sum = 0
        epsilon = 1e-9 # Prevents log(0) errors

        # 3. Iterate through every segment defined by the boundaries
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            
            # Constraint check: if points are too close, penalise heavily
            if end <= start: 
                return np.inf
            
            # Get stats for this segment
            k, n = self._get_bin_stats(start, end)
            
            # If a bin is empty, it contributes nothing to the likelihood
            if n == 0:
                continue
                
            p = k / n
            
            # Clip probabilities to avoid log(0)
            p = np.clip(p, epsilon, 1 - epsilon)
            
            # MLE Formula: k * ln(p) + (n-k) * ln(1-p)
            segment_ll = k * np.log(p) + (n - k) * np.log(1 - p)
            log_likelihood_sum += segment_ll

        # Return negative because 'minimize' tries to find the smallest value
        return -log_likelihood_sum

    def optimize_segmentation(self, num_points=10):
        # Initial guess: evenly spaced points
        initial_guess = np.linspace(350, 800, num_points)
        
        # Run Optimisation
        # I use 'Powell' because it is a derivative-free method.
        # It handles the "step-function" nature of integer binning much better
        # than L-BFGS-B.
        result = minimize(
            self.negative_log_likelihood, 
            initial_guess, 
            method='Powell'
        )
        
        return result

# --- Execution ---

# 1. Initialise
optimiser = DefaultLikelihoodOptimizer('Loan_Data.csv')

# 2. Run Optimisation
result = optimiser.optimize_segmentation(num_points=5)

# 3. Print Results
print(f"Maximized Log-Likelihood: {-result.fun:.2f}")
print("Optimal FICO Thresholds:")
print(sorted([int(p) for p in result.x]))
