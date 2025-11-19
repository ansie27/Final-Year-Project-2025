import numpy as np

def calculate_lambda_max(comparison_matrix: np.ndarray, 
                         weights: np.ndarray) -> float:
    """
    Calculate λ_max (maximum eigenvalue) for consistency checking.
    
    Formula: λ_max = Σ(weighted_sum_i / weight_i) / n
    """

def calculate_consistency_index(comparison_matrix: np.ndarray, 
                                weights: np.ndarray) -> float:
    """
    Calculate CI per Equation 6: CI = (λ_max - n) / (n - 1)
    """

def get_random_index(n: int) -> float:
    """
    Lookup Random Index (RI) for consistency ratio calculation.
    
    RI values from Saaty (2008):
    n:  1    2    3    4    5    6    7    8    9    10
    RI: 0    0   0.58 0.90 1.12 1.24 1.32 1.41 1.45 1.49
    """

def calculate_consistency_ratio(comparison_matrix: np.ndarray, 
                                weights: np.ndarray) -> float:
    """
    Calculate CR per Equation 7: CR = CI / RI
    """

def is_consistent(comparison_matrix: np.ndarray, 
                  weights: np.ndarray, 
                  threshold: float = 0.1) -> tuple[bool, float]:
    """
    Check if pairwise comparisons are consistent.
    
    Returns:
        (is_consistent, CR_value)
    
    Threshold: CR < 0.1 is acceptable (Saaty standard)
    """

def suggest_improvements(comparison_matrix: np.ndarray, 
                         weights: np.ndarray) -> dict:
    """
    Identify inconsistent pairwise comparisons and suggest adjustments.
    """