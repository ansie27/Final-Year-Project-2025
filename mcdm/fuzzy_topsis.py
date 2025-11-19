import numpy as np
import pandas as pd

class FuzzyTOPSIS:
    def __init__(self, 
                 decision_matrix: pd.DataFrame,
                 criteria_types: dict[str, str],
                 fuzzy_weights: np.ndarray = None,
                 crisp_weights: np.ndarray = None):
        """
        Initialize Fuzzy TOPSIS.
        
        Parameters:
            decision_matrix: Alternatives x Criteria dataframe
            criteria_types: {'criterion_name': 'benefit' or 'cost'}
            fuzzy_weights: n x 3 array of TFNs (if using fuzzy)
            crisp_weights: n array (if using crisp from defuzzified AHP)
        """
        self.decision_matrix = decision_matrix
        self.criteria_types = criteria_types
        self.fuzzy_weights = fuzzy_weights
        self.crisp_weights = crisp_weights
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.fpis = None  # Fuzzy Positive Ideal Solution
        self.fnis = None  # Fuzzy Negative Ideal Solution
        self.scores = None
        
    def create_fuzzy_decision_matrix(self) -> np.ndarray:
        """
        Convert crisp decision matrix to fuzzy ratings.
        
        Options:
        1. Use linguistic terms → TFNs (if input is categorical)
        2. Use alpha-cuts to create TFNs from crisp values
        3. Expert fuzzy ratings (if available)
        
        Returns:
            mxnx3 array (alternatives x criteria x fuzzy bounds)
        """
        
    def normalize_fuzzy_matrix(self) -> np.ndarray:
        """
        Normalize fuzzy decision matrix per Equations 17.1, 17.2.
        
        For benefit criteria: r̃_ij = (a_ij/c_j*, b_ij/c_j*, c_ij/c_j*)
        For cost criteria: r̃_ij = (a_j-/c_ij, a_j-/b_ij, a_j-/a_ij)
        """
        
    def apply_fuzzy_weights(self) -> np.ndarray:
        """
        Calculate weighted normalized fuzzy matrix per Equation 18.
        
        ṽ_ij = w̃_j ⊗ r̃_ij (fuzzy multiplication)
        """
        
    def determine_fpis_fnis(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate FPIS and FNIS per Equations 19.1, 19.2.
        
        FPIS: A* = (ṽ_1*, ṽ_2*, ..., ṽ_n*)
        FNIS: A- = (ṽ_1-, ṽ_2-, ..., ṽ_n-)
        
        Returns:
            (fpis, fnis) - both nx3 arrays
        """
        
    def calculate_fuzzy_distances(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate distances to FPIS and FNIS per Equations 20.1, 20.2.
        
        Uses vertex method or Euclidean distance for TFNs.
        
        Returns:
            (d_i*, d_i-) - distances to ideal and anti-ideal
        """
        
    def calculate_closeness_coefficients(self) -> np.ndarray:
        """
        Calculate CC per Equation 21: CC_i = d_i- / (d_i* + d_i-)
        
        Returns:
            Array of closeness coefficients [0, 1]
            Higher = closer to ideal solution
        """
        
    def rank_alternatives(self) -> pd.DataFrame:
        """
        Rank alternatives by CC in descending order.
        """
        
    def fit(self) -> 'FuzzyTOPSIS':
        """
        Complete Fuzzy TOPSIS pipeline:
        1. Create/normalize fuzzy matrix
        2. Apply weights
        3. Determine FPIS/FNIS
        4. Calculate distances
        5. Compute closeness coefficients
        6. Rank alternatives
        """
        
    def get_rankings(self) -> pd.DataFrame:
        """Return final rankings with scores."""