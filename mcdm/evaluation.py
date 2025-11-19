from fuzzy_ahp_topsis_ga import FuzzyAHPTOPSISGA
import pandas as pd

class FuzzyAHPTOPSISEvaluator:
    def __init__(self, model: FuzzyAHPTOPSISGA):
        self.model = model
        
    def evaluate_consistency(self) -> dict:
        """
        Evaluate AHP consistency per Equations 6-7.
        
        Returns:
            {
                'CI': float,
                'CR': float,
                'is_consistent': bool,
                'interpretation': str
            }
        """
        
    def evaluate_convergence(self) -> dict:
        """
        Evaluate GA convergence per Phase 6.
        
        Checks:
        - Did GA reach stopping criterion?
        - Fitness improvement over generations
        - Population diversity
        """
        
    def evaluate_sensitivity(self, 
                            perturbation_levels: list = [0.05, 0.10, 0.15]) -> pd.DataFrame:
        """
        Sensitivity analysis per Phase 6 evaluation criteria.
        
        Tests ranking stability under weight perturbations.
        """
        
    def evaluate_ranking_quality(self) -> dict:
        """
        Assess quality of final rankings.
        
        Metrics:
        - Score separation (variance)
        - Ranking stability
        - Kendall's tau (if reference available)
        """
        
    def generate_evaluation_report(self) -> dict:
        """
        Comprehensive evaluation report for Phase 6.
        
        Includes all metrics required by documentation.
        """