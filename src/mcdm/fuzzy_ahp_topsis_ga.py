import pandas as pd
import numpy as np

class FuzzyAHPTOPSISGA:
    def __init__(self, 
                 criteria: list[str],
                 criteria_types: dict[str, str],
                 expert_judgments: dict = None,
                 use_ga_optimization: bool = True,
                 use_local_search: bool = True,
                 integration_strategy: str = 'hybrid'):
        """
        Complete Fuzzy AHP-TOPSIS-GA framework.
        
        Parameters:
            criteria: List of criterion names
            criteria_types: {'criterion': 'benefit'/'cost'}
            expert_judgments: Optional pairwise comparisons
            use_ga_optimization: Enable GA per Phase 5
            use_local_search: Enable local search per Phase 5
            integration_strategy: 'sequential', 'parallel', 'hybrid'
        """
        self.criteria = criteria
        self.criteria_types = criteria_types
        self.expert_judgments = expert_judgments
        self.use_ga = use_ga_optimization
        self.use_local_search = use_local_search
        self.strategy = integration_strategy
        
        # Components
        self.fuzzy_ahp = None
        self.fuzzy_topsis = None
        self.ga_optimizer = None
        self.local_search = None
        
        # Results
        self.final_weights = None
        self.final_rankings = None
        self.evaluation_metrics = {}
        
    def fit(self, decision_matrix: pd.DataFrame) -> 'FuzzyAHPTOPSISGA':
        """
        Execute complete pipeline per Phase 5 workflow:
        
        1. Fuzzy AHP for initial weights
        2. Check AHP consistency (Phase 6 requirement)
        3. GA optimization (if enabled)
        4. Fuzzy TOPSIS with optimized weights
        5. Local search refinement (if enabled)
        6. Final ranking generation
        """
        
    def _phase_1_fuzzy_ahp(self) -> np.ndarray:
        """Step 1: Fuzzy AHP analysis."""
        
    def _phase_2_consistency_check(self) -> dict:
        """Step 2: Validate consistency per Equations 6-7."""
        
    def _phase_3_ga_optimization(self) -> np.ndarray:
        """Step 3: GA weight optimization (optional)."""
        
    def _phase_4_fuzzy_topsis(self, decision_matrix: pd.DataFrame) -> pd.DataFrame:
        """Step 4: Fuzzy TOPSIS ranking."""
        
    def _phase_5_local_search(self) -> pd.DataFrame:
        """Step 5: Local search refinement (optional)."""
        
    def get_rankings(self) -> pd.DataFrame:
        """Return final supplier/commodity rankings."""
        
    def get_weights(self, weight_type: str = 'final') -> np.ndarray:
        """
        Get weights at different pipeline stages.
        
        Parameters:
            weight_type: 'initial_ahp', 'ga_optimized', 'final'
        """
        
    def get_evaluation_metrics(self) -> dict:
        """
        Return evaluation metrics per Phase 6:
        - Consistency Index (CI)
        - Consistency Ratio (CR)
        - GA convergence status
        - Local search improvement
        """
        
    def sensitivity_analysis(self, 
                            perturbation_range: float = 0.1,
                            n_iterations: int = 100) -> pd.DataFrame:
        """
        Perform sensitivity analysis per Phase 6 evaluation.
        
        Perturb weights and measure ranking stability.
        """
        
    def export_results(self, 
                       output_dir: str = './outputs/fuzzy_ahp_topsis',
                       format: str = 'all') -> None:
        """
        Export results for Phase 7 dashboard integration.
        
        Formats: 'csv', 'json', 'excel', 'all'
        """