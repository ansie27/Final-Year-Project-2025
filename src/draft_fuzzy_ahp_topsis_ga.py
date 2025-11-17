"""
Fuzzy AHP-TOPSIS Algorithm Optimized with Genetic Algorithm (GA)

This module implements:
1. Fuzzy Analytic Hierarchy Process (AHP) for criteria weighting
2. Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) for ranking
3. Genetic Algorithm (GA) for optimization of criteria weights
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

class FuzzyNumber:
    def __init__(self, l, m, u):
        # Triangular fuzzy number (l, m, u)
        self.l = l  # Lower bound
        self.m = m  # Most likely value
        self.u = u  # Upper bound
    
    def __repr__(self):
        return f"FuzzyNumber({self.l}, {self.m}, {self.u})"
    
    def defuzzify(self):
        # Convert fuzzy number to crisp value with centroid method
        return (self.l + 2 * self.m + self.u) / 4


# Fuzzy pairwise comparison matrix for AHP
def create_fuzzy_comparison_matrix(criteria_count, random_state=None): # Should be consistent?
    if random_state is not None:
        np.random.seed(random_state)
    
    n = criteria_count
    # Initialize matrix with fuzzy numbers
    matrix = np.zeros((n, n, 3))
    
    # Linguistic scale for fuzzy AHP (triangular fuzzy numbers)
    # Referenced Abimbola et al. (2020)
    fuzzy_scale = {
        1: FuzzyNumber(1, 1, 1),      # Equal importance
        2: FuzzyNumber(1, 2, 3),      # Equally to moderate importance
        3: FuzzyNumber(2, 3, 4),      # Moderate importance
        4: FuzzyNumber(3, 4, 5),      # Moderately to strong importance
        5: FuzzyNumber(4, 5, 6),      # Strong importance
        6: FuzzyNumber(5, 6, 7),      # Strongly to very strong importance
        7: FuzzyNumber(6, 7, 8),      # Very strongly importance
        8: FuzzyNumber(7, 8, 9),      # Very strongly to extremely important
        9: FuzzyNumber(8, 9, 9)       # Extremely important 
    }
    
    # Fill diagonal with (1, 1, 1)
    for i in range(n):
        matrix[i, i] = [1, 1, 1]
    
    # Fill upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            # Randomly select importance level (1-9)
            importance = np.random.randint(1, 10)
            fuzzy_val = fuzzy_scale[importance]
            matrix[i, j] = [fuzzy_val.l, fuzzy_val.m, fuzzy_val.u]
            # Reciprocal for lower triangle
            matrix[j, i] = [1/fuzzy_val.u, 1/fuzzy_val.m, 1/fuzzy_val.l]
    
    return matrix

# Calculate fuzzy AHP weights from the comparison matrix
def fuzzy_ahp_weights(fuzzy_matrix):
    n = fuzzy_matrix.shape[0]
    
    # Calculate geometric mean for each row
    geometric_means = np.zeros((n, 3))
    
    for i in range(n):
        l_prod = 1
        m_prod = 1
        u_prod = 1
        
        for j in range(n):
            l_prod *= fuzzy_matrix[i, j, 0]
            m_prod *= fuzzy_matrix[i, j, 1]
            u_prod *= fuzzy_matrix[i, j, 2]
        
        geometric_means[i, 0] = l_prod ** (1/n)
        geometric_means[i, 1] = m_prod ** (1/n)
        geometric_means[i, 2] = u_prod ** (1/n)
    
    # Sum of geometric means
    sum_l = np.sum(geometric_means[:, 0])
    sum_m = np.sum(geometric_means[:, 1])
    sum_u = np.sum(geometric_means[:, 2])
    
    # Normalize fuzzy weights
    fuzzy_weights = np.zeros((n, 3))
    for i in range(n):
        fuzzy_weights[i, 0] = geometric_means[i, 0] / sum_u
        fuzzy_weights[i, 1] = geometric_means[i, 1] / sum_m
        fuzzy_weights[i, 2] = geometric_means[i, 2] / sum_l
    
    # Defuzzify to get crisp weights
    crisp_weights = np.array([FuzzyNumber(w[0], w[1], w[2]).defuzzify() for w in fuzzy_weights])
    
    # Normalize to sum to 1
    crisp_weights = crisp_weights / np.sum(crisp_weights)
    
    return crisp_weights

# Normalise the decision matrix for TOPSIS
def normalize_decision_matrix(decision_matrix, criteria_types):
    n_alternatives, n_criteria = decision_matrix.shape
    normalized = np.zeros_like(decision_matrix)
    
    for j in range(n_criteria):
        column = decision_matrix[:, j]
        
        # Vector normalization
        norm = np.sqrt(np.sum(column ** 2))
        if norm > 0:
            normalized[:, j] = column / norm
        else:
            normalized[:, j] = column
    
    return normalized

def calculate_topsis_scores(decision_matrix, weights, criteria_types):
    # Normalize decision matrix
    normalized = normalize_decision_matrix(decision_matrix, criteria_types)
    
    # Weighted normalized matrix
    weighted_normalized = normalized * weights
    
    # Determine ideal and negative ideal solutions
    ideal_solution = np.zeros(len(weights))
    negative_ideal_solution = np.zeros(len(weights))
    
    for j in range(len(weights)):
        if criteria_types[j] == 'benefit':
            ideal_solution[j] = np.max(weighted_normalized[:, j])
            negative_ideal_solution[j] = np.min(weighted_normalized[:, j])
        else:  # cost
            ideal_solution[j] = np.min(weighted_normalized[:, j])
            negative_ideal_solution[j] = np.max(weighted_normalized[:, j])
    
    # Calculate distances
    n_alternatives = decision_matrix.shape[0]
    distances_ideal = np.zeros(n_alternatives)
    distances_negative_ideal = np.zeros(n_alternatives)
    
    for i in range(n_alternatives):
        distances_ideal[i] = np.sqrt(np.sum((weighted_normalized[i] - ideal_solution) ** 2))
        distances_negative_ideal[i] = np.sqrt(np.sum((weighted_normalized[i] - negative_ideal_solution) ** 2))
    
    # Calculate relative closeness
    scores = distances_negative_ideal / (distances_ideal + distances_negative_ideal + 1e-10)
    
    return scores


def genetic_algorithm_optimization(decision_matrix, criteria_types, 
                                  population_size=50, generations=100, 
                                  mutation_rate=0.1, crossover_rate=0.8,
                                  random_state=None):
    """
    Optimize criteria weights using Genetic Algorithm.
    
    Parameters:
    -----------
    decision_matrix : np.ndarray
        Decision matrix (alternatives x criteria)
    criteria_types : list
        List of 'benefit' or 'cost' for each criterion
    population_size : int
        Size of GA population
    generations : int
        Number of GA generations
    mutation_rate : float
        Mutation rate
    crossover_rate : float
        Crossover rate
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    dict
        Optimization results including best weights and scores
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_criteria = decision_matrix.shape[1]
    
    def objective_function(weights):
        """Objective: maximize separation between alternatives."""
        # Ensure weights sum to 1
        weights = np.abs(weights)
        weights = weights / (np.sum(weights) + 1e-10)
        
        # Calculate TOPSIS scores
        scores = calculate_topsis_scores(decision_matrix, weights, criteria_types)
        
        # Objective: maximize variance (better separation)
        # Negative because we're minimizing
        return -np.var(scores)
    
    # Bounds for weights (0 to 1 for each criterion)
    bounds = [(0, 1) for _ in range(n_criteria)]
    
    # Use differential evolution (a type of GA)
    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=generations,
        popsize=population_size,
        mutation=mutation_rate,
        recombination=crossover_rate,
        seed=random_state,
        polish=False
    )
    
    # Normalize final weights
    best_weights = np.abs(result.x)
    best_weights = best_weights / np.sum(best_weights)
    
    # Calculate final TOPSIS scores with optimized weights
    final_scores = calculate_topsis_scores(decision_matrix, best_weights, criteria_types)
    
    return {
        'optimized_weights': best_weights,
        'topsis_scores': final_scores,
        'optimization_success': result.success,
        'optimization_message': result.message,
        'objective_value': -result.fun
    }


def fuzzy_ahp_topsis_ga(data, criteria_columns, criteria_types, 
                        supplier_id_column='Supplier_ID',
                        use_ga_optimization=True,
                        ga_params=None,
                        random_state=42):
    """
    Complete Fuzzy AHP-TOPSIS-GA analysis for supplier ranking.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with supplier data
    criteria_columns : list
        List of column names to use as criteria
    criteria_types : list
        List of 'benefit' or 'cost' for each criterion
    supplier_id_column : str
        Column name for supplier IDs
    use_ga_optimization : bool
        Whether to use GA for weight optimization
    ga_params : dict, optional
        GA parameters (population_size, generations, etc.)
    random_state : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Suppliers ranked by TOPSIS scores
    """
    np.random.seed(random_state)
    
    # Prepare decision matrix
    decision_matrix = data[criteria_columns].values
    n_criteria = len(criteria_columns)
    
    # Handle missing values
    decision_matrix = np.nan_to_num(decision_matrix, nan=0.0)
    
    # Normalize criteria types
    if len(criteria_types) != n_criteria:
        # Default: all benefit criteria
        criteria_types = ['benefit'] * n_criteria
    
    # Step 1: Fuzzy AHP for initial weights
    print("  [1/3] Performing Fuzzy AHP analysis...")
    fuzzy_matrix = create_fuzzy_comparison_matrix(n_criteria, random_state=random_state)
    initial_weights = fuzzy_ahp_weights(fuzzy_matrix)
    print(f"     ✓ Initial weights calculated using Fuzzy AHP")
    
    # Step 2: GA Optimization (optional)
    if use_ga_optimization:
        print("  [2/3] Optimizing weights using Genetic Algorithm...")
        if ga_params is None:
            ga_params = {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            }
        
        ga_results = genetic_algorithm_optimization(
            decision_matrix,
            criteria_types,
            random_state=random_state,
            **ga_params
        )
        optimized_weights = ga_results['optimized_weights']
        topsis_scores = ga_results['topsis_scores']
        print(f"     ✓ GA optimization completed")
        print(f"     ✓ Objective value: {ga_results['objective_value']:.4f}")
    else:
        print("  [2/3] Using Fuzzy AHP weights (GA optimization skipped)...")
        optimized_weights = initial_weights
        topsis_scores = calculate_topsis_scores(decision_matrix, optimized_weights, criteria_types)
    
    # Step 3: TOPSIS ranking
    print("  [3/3] Calculating TOPSIS scores and ranking suppliers...")
    
    # Create results dataframe
    results = data[[supplier_id_column]].copy()
    results['Fuzzy_AHP_Weights'] = [initial_weights] * len(results)
    results['Optimized_Weights'] = [optimized_weights] * len(results)
    results['TOPSIS_Score'] = topsis_scores
    results['Rank'] = results['TOPSIS_Score'].rank(ascending=False, method='min').astype(int)
    
    # Sort by rank
    results = results.sort_values('Rank')
    
    print(f"     ✓ {len(results)} suppliers ranked")
    
    return results


def analyze_supplier_ranking(data, criteria_columns, criteria_types,
                            supplier_id_column='Supplier_ID',
                            top_n=10,
                            use_ga_optimization=True,
                            random_state=42):
    """
    Analyze and rank suppliers using Fuzzy AHP-TOPSIS-GA.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with supplier data
    criteria_columns : list
        List of column names to use as criteria
    criteria_types : list
        List of 'benefit' or 'cost' for each criterion
    supplier_id_column : str
        Column name for supplier IDs
    top_n : int
        Number of top suppliers to display
    use_ga_optimization : bool
        Whether to use GA optimization
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Analysis results including rankings and weights
    """
    print("\n" + "="*70)
    print("FUZZY AHP-TOPSIS-GA SUPPLIER RANKING ANALYSIS")
    print("="*70)
    
    # Perform analysis
    ranking_results = fuzzy_ahp_topsis_ga(
        data,
        criteria_columns,
        criteria_types,
        supplier_id_column=supplier_id_column,
        use_ga_optimization=use_ga_optimization,
        random_state=random_state
    )
    
    # Display top N suppliers
    print(f"\nTop {top_n} Ranked Suppliers:")
    print("-" * 70)
    top_suppliers = ranking_results.head(top_n)
    for idx, row in top_suppliers.iterrows():
        print(f"Rank {int(row['Rank'])}: {row[supplier_id_column]} - "
              f"TOPSIS Score: {row['TOPSIS_Score']:.4f}")
    
    # Weight information
    if len(ranking_results) > 0:
        sample_weights = ranking_results.iloc[0]['Optimized_Weights']
        print(f"\nOptimized Criteria Weights:")
        print("-" * 70)
        for i, (col, weight) in enumerate(zip(criteria_columns, sample_weights)):
            print(f"  {col}: {weight:.4f}")
    
    print("\n" + "="*70)
    
    return {
        'ranking_results': ranking_results,
        'criteria_columns': criteria_columns,
        'criteria_types': criteria_types
    }


if __name__ == "__main__":
    print("Fuzzy AHP-TOPSIS-GA Module")
    print("Import this module to use Fuzzy AHP-TOPSIS-GA functions in your pipeline.")