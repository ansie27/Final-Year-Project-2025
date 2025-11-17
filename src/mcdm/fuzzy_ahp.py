import numpy as np

class FuzzyAHP:
    def __init__(self, criteria: list[str], 
                 expert_judgments: dict = None,
                 random_state: int = 42):
        """
        Initialize Fuzzy AHP with criteria and optional expert judgments.
        
        Parameters:
            criteria: List of criterion names
            expert_judgments: Dict of {('C1', 'C2'): fuzzy_value, ...}
                             If None, will prompt or use defaults
        """
        self.criteria = criteria
        self.n_criteria = len(criteria)
        self.expert_judgments = expert_judgments
        self.fuzzy_matrix = None
        self.fuzzy_weights = None
        self.crisp_weights = None
        self.consistency_ratio = None
        
    def create_fuzzy_comparison_matrix(self) -> np.ndarray:
        """
        Create fuzzy pairwise comparison matrix per Equation 4.
        Uses expert_judgments or Table 3 priorities, NOT random.
        
        Returns:
            nxnx3 array (lower, middle, upper bounds)
        """
        
    def _load_default_judgments(self) -> dict:
        """
        Load domain knowledge from Table 3 for green supplier criteria.
        
        Priority hierarchy (from documentation):
        1. Environmental: ISO 14001 > GHG emissions > waste mgmt
        2. Quality: ISO 9001 > defect rate
        3. Cost: total cost > unit price
        4. Delivery: on-time > lead time
        """
        
    def calculate_fuzzy_synthetic_extent(self) -> np.ndarray:
        """
        Calculate fuzzy synthetic extent per Equation 8.
        
        Formula: S_i = Σ M_ij ⊗ [Σ Σ M_ij]^(-1)
        """
        
    def calculate_fuzzy_weights(self) -> np.ndarray:
        """
        Derive fuzzy priority weights from comparison matrix.
        
        Steps:
        1. Calculate geometric mean of each row (fuzzy)
        2. Sum all geometric means
        3. Normalize: weight_i = geom_mean_i / sum_geom_means
        
        Returns:
            nx3 array of fuzzy weights
        """
        
    def defuzzify_weights(self) -> np.ndarray:
        """
        Convert fuzzy weights to crisp values and normalize to sum=1.
        """
        
    def check_consistency(self) -> dict:
        """
        Validate pairwise comparison consistency per Equations 6-7.
        
        Returns:
            {
                'CI': float,
                'CR': float,
                'is_consistent': bool,
                'lambda_max': float
            }
        """
        
    def fit(self) -> 'FuzzyAHP':
        """
        Complete Fuzzy AHP pipeline:
        1. Create comparison matrix
        2. Calculate fuzzy weights
        3. Defuzzify
        4. Check consistency
        """
        
    def get_weights(self) -> np.ndarray:
        """Return final crisp weights."""
        
    def visualize_hierarchy(self) -> None:
        """Plot criteria importance hierarchy."""