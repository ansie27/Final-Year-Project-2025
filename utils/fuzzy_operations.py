from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

SAATY_SCALE_VALUES: Dict[int, Tuple[float, float, float]] = {
    1: (1.0, 1.0, 1.0),
    2: (1.0, 2.0, 3.0),
    3: (2.0, 3.0, 4.0),
    4: (3.0, 4.0, 5.0),
    5: (4.0, 5.0, 6.0),
    6: (5.0, 6.0, 7.0),
    7: (6.0, 7.0, 8.0),
    8: (7.0, 8.0, 9.0),
    9: (8.0, 9.0, 9.0),
}

def _to_tfn(value: float | "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":
    if isinstance(value, TriangularFuzzyNumber):
        return value
    if isinstance(value, (int, float)):
        return TriangularFuzzyNumber(float(value), float(value), float(value))
    raise TypeError("Value must be a TriangularFuzzyNumber or numeric scalar.")

@dataclass(frozen=True)
class TriangularFuzzyNumber:
    l: float
    m: float
    u: float

    def __post_init__(self) -> None:
        if not (self.l <= self.m <= self.u):
            raise ValueError("Triangular fuzzy numbers require l <= m <= u.")

    # Fuzzy addition
    def __add__(self, other: float | "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":
        other_tfn = _to_tfn(other)
        return TriangularFuzzyNumber(self.l + other_tfn.l, self.m + other_tfn.m, self.u + other_tfn.u)

    __radd__ = __add__

    # Fuzzy multiplication
    def __mul__(self, other: float | "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":
        other_tfn = _to_tfn(other)
        products = sorted(
            a * b
            for a in (self.l, self.m, self.u)
            for b in (other_tfn.l, other_tfn.m, other_tfn.u)
        )
        middle = products[len(products) // 2]
        return TriangularFuzzyNumber(products[0], middle, products[-1])

    __rmul__ = __mul__

    # Fuzzy division
    def __truediv__(self, other: float | "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":
        other_tfn = _to_tfn(other)
        return self * other_tfn.reciprocal()

    def __rtruediv__(self, other: float | "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":
        return _to_tfn(other) * self.reciprocal()

    # Reciprocals
    def reciprocal(self) -> "TriangularFuzzyNumber":
        if self.l <= 0:
            raise ValueError("Reciprocal is only defined for positive fuzzy numbers.")
        return TriangularFuzzyNumber(1 / self.u, 1 / self.m, 1 / self.l)

    @classmethod
    def from_saaty_scale(cls, level: int) -> "TriangularFuzzyNumber":
        try:
            l, m, u = SAATY_SCALE_VALUES[level]
        except KeyError:
            raise ValueError("Saaty scale level must be an integer between 1 and 9.")
        return cls(l, m, u)

    # Defuzzification with the centroid method
    def defuzzify(self) -> float:
        return (self.l + self.m + self.u) / 3.0

    # Fuzzy distance
    def distance_to(self, other: float | "TriangularFuzzyNumber", method: str = "vertex") -> float:
        other_tfn = _to_tfn(other)
        metric = method.lower()
        if metric == "vertex":
            return vertex_distance(self, other_tfn)
        if metric == "euclidean":
            return euclidean_distance(self, other_tfn)
        if metric == "hamming":
            return hamming_distance(self, other_tfn)
        raise ValueError(f"Unsupported distance method: {method}")

    @staticmethod
    def from_linguistic(term: str) -> "TriangularFuzzyNumber":
        scale: Dict[str, Tuple[float, float, float]] = {
            "very low": (0.0, 0.0, 0.25),
            "low": (0.0, 0.25, 0.5),
            "medium": (0.25, 0.5, 0.75),
            "high": (0.5, 0.75, 1.0),
            "very high": (0.75, 1.0, 1.0),
        }
        key = term.strip().lower()
        if key not in scale:
            raise ValueError(f"Unknown linguistic term '{term}'. Supported terms: {', '.join(scale)}.")
        l, m, u = scale[key]
        return TriangularFuzzyNumber(l, m, u)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.l, self.m, self.u)

    def to_array(self) -> np.ndarray:
        return np.array(self.to_tuple(), dtype=float)

    def __repr__(self) -> str:
        return f"TFN({self.l:.3f}, {self.m:.3f}, {self.u:.3f})"


def _normalise_inputs(
    a: float | TriangularFuzzyNumber,
    b: float | TriangularFuzzyNumber,
) -> tuple[TriangularFuzzyNumber, TriangularFuzzyNumber]:
    return _to_tfn(a), _to_tfn(b)


def vertex_distance(
    a: float | TriangularFuzzyNumber,
    b: float | TriangularFuzzyNumber,
) -> float:
    first, second = _normalise_inputs(a, b)
    diff = first.to_array() - second.to_array()
    return float(np.sqrt(np.mean(diff**2)))


def euclidean_distance(
    a: float | TriangularFuzzyNumber,
    b: float | TriangularFuzzyNumber,
) -> float:
    first, second = _normalise_inputs(a, b)
    diff = first.to_array() - second.to_array()
    return float(np.linalg.norm(diff))


def hamming_distance(
    a: float | TriangularFuzzyNumber,
    b: float | TriangularFuzzyNumber,
) -> float:
    first, second = _normalise_inputs(a, b)
    diff = np.abs(first.to_array() - second.to_array())
    return float(np.mean(diff))


def calculate_lambda_max(comparison_matrix: np.ndarray, weights: np.ndarray) -> float:
    matrix = np.asarray(comparison_matrix, dtype=float)
    w = np.asarray(weights, dtype=float).flatten()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Comparison matrix must be square.")
    if matrix.shape[0] != w.shape[0]:
        raise ValueError("Weights vector length must match comparison matrix size.")
    weighted_sum = matrix @ w
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(weighted_sum, w, out=np.zeros_like(weighted_sum), where=w != 0)
    return float(np.mean(ratios))


def calculate_consistency_index(comparison_matrix: np.ndarray, weights: np.ndarray) -> float:
    matrix = np.asarray(comparison_matrix, dtype=float)
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    lambda_max = calculate_lambda_max(matrix, weights)
    return float((lambda_max - n) / (n - 1))

def get_random_index(n: int) -> float:
    random_index = {
        1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
        11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
    }
    
    if n < 1:
        raise ValueError("n must be positive")
    
    # Return exact value if available, otherwise use n=11 approximation
    return random_index.get(n, 1.51)

# CR < 0.1 = acceptable
# CR 0.05 for n=3, CR 0.08 for n=4
def calculate_consistency_ratio(comparison_matrix: np.ndarray, weights: np.ndarray) -> float:
    matrix = np.asarray(comparison_matrix, dtype=float)
    n = matrix.shape[0]
    
    # For n < 3, CR is not applicable
    if n < 3:
        return 0.0
    
    ci = calculate_consistency_index(matrix, weights)

    random_index = {
        1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
        11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
    }
    
    ri = random_index.get(n, 1.51)  # Default to n=11
    
    if np.isclose(ri, 0.0):
        return 0.0
    
    cr = ci / ri
    
    # If CR is poor
    if cr > 0.10:
        import warnings
        warnings.warn(
            f"Consistency Ratio ({cr:.4f}) exceeds 0.10 threshold. "
            "Consider revising pairwise comparisons.",
            UserWarning
        )
    
    return float(cr)

def is_consistent(
    comparison_matrix: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.1,
) -> tuple[bool, float]:
    cr = calculate_consistency_ratio(comparison_matrix, weights)
    return cr < threshold, cr


def suggest_improvements(
    comparison_matrix: np.ndarray,
    weights: np.ndarray,
    tolerance: float = 0.1,
) -> dict:
    matrix = np.asarray(comparison_matrix, dtype=float)
    w = np.asarray(weights, dtype=float).flatten()
    expected = np.divide.outer(w, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = np.abs(matrix - expected) / expected
    suggestions = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isfinite(deviation[i, j]):
                continue
            if deviation[i, j] > tolerance:
                suggestions.append(
                    {
                        "pair": (i, j),
                        "current": float(matrix[i, j]),
                        "suggested": float(expected[i, j]),
                        "deviation": float(deviation[i, j]),
                    }
                )
    return {"tolerance": tolerance, "inconsistent_pairs": suggestions}