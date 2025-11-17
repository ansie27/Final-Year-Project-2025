import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")

# -------- FUZZY AHP CLASS -----------
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

# -------- FUZZY TOPSIS CLASS -----------

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

# -------- GENETIC ALGORITHM OPTIMISER CLASS -----------
class GeneticAlgorithmOptimizer:
    def __init__(self,
                 fuzzy_ahp: FuzzyAHP,
                 fuzzy_topsis: FuzzyTOPSIS,
                 integration_mode: str = 'ahp_optimization'):
        """
        Initialize GA optimizer.
        
        Parameters:
            fuzzy_ahp: FuzzyAHP instance
            fuzzy_topsis: FuzzyTOPSIS instance
            integration_mode: 
                'ahp_optimization' - optimize fuzzy comparison matrix
                'weight_tuning' - directly optimize final weights
                'hybrid' - combine AHP and data-driven weights
        """
        self.ahp = fuzzy_ahp
        self.topsis = fuzzy_topsis
        self.mode = integration_mode
        
    def _initialize_population(self, pop_size: int) -> np.ndarray:
        """
        Create initial population of weight vectors.
        Ensure each individual sums to 1.
        """
        
    def _fitness_function(self, individual: np.ndarray) -> float:
        """
        Evaluate fitness of a weight vector.
        
        Options (per documentation):
        1. Maximize TOPSIS score variance (better discrimination)
        2. Maximize consistency with AHP structure
        3. Multi-objective: balance both
        """
        
    def _selection(self, population: np.ndarray, 
                   fitness: np.ndarray,
                   method: str = 'tournament') -> np.ndarray:
        """
        Select parents for reproduction.
        Methods: 'roulette', 'tournament', 'rank'
        """
        
    def _crossover(self, parent1: np.ndarray, 
                   parent2: np.ndarray,
                   method: str = 'simulated_binary') -> tuple:
        """
        Crossover operation maintaining sum=1 constraint.
        """
        
    def _mutation(self, individual: np.ndarray, 
                  mutation_rate: float) -> np.ndarray:
        """
        Mutation operation with constraint preservation.
        """
        
    def _elitism(self, population: np.ndarray, 
                 fitness: np.ndarray,
                 n_elite: int) -> np.ndarray:
        """
        Preserve top N individuals across generations.
        """
        
    def optimize(self,
                 population_size: int = 50,
                 n_generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_count: int = 2,
                 random_state: int = 42) -> dict:
        """
        Run genetic algorithm optimization per Section 3.5.1.
        
        Returns:
            {
                'best_weights': np.ndarray,
                'best_fitness': float,
                'generation_history': list,
                'convergence_achieved': bool
            }
        """
        
    def hybrid_weight_integration(self, 
                                  alpha: float = 0.5) -> np.ndarray:
        """
        Combine AHP weights and GA weights per improvement suggestion:
        
        final_weight = alpha x fuzzy_ahp_weight + (1-alpha) x ga_optimized_weight
        
        alpha ∈ [0,1] balances expert knowledge vs data-driven optimization
        """

# -------- FUZZY AHP-TOPSIS WITH OPTIMISATION ------------
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

# ------- EVALUATION -----------
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