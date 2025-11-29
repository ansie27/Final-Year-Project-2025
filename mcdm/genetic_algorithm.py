from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


FitnessFunction = Callable[[np.ndarray], float]


@dataclass
class GAParameters:
    generations: int = 60
    population_size: int = 40
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    tournament_size: int = 3
    random_state: int = 42


class WeightGAOptimizer:
    """
    Lightweight GA helper that can be reused by Fuzzy AHP and TOPSIS workflows.
    Expects a fitness function that the optimizer will minimise.
    """

    def __init__(self, params: Optional[GAParameters] = None) -> None:
        self.params = params or GAParameters()
        self.rng = np.random.default_rng(self.params.random_state)

    def _initialise_population(self, seed: np.ndarray) -> np.ndarray:
        n = len(seed)
        pop = self.rng.random((self.params.population_size, n)) + 1e-6
        pop[0] = seed
        return self._normalise(pop)

    def _normalise(self, population: np.ndarray) -> np.ndarray:
        totals = population.sum(axis=1, keepdims=True)
        totals = np.where(np.isclose(totals, 0), 1e-12, totals)
        return population / totals

    def _tournament_select(self, population: np.ndarray, scores: np.ndarray) -> np.ndarray:
        idx = self.rng.choice(
            len(population),
            size=min(self.params.tournament_size, len(population)),
            replace=False,
        )
        best_idx = idx[np.argmin(scores[idx])]
        return population[best_idx]

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        alpha = self.rng.random()
        child = alpha * parent_a + (1 - alpha) * parent_b
        return child

    def _mutate(self, candidate: np.ndarray) -> np.ndarray:
        noise = self.rng.lognormal(mean=0.0, sigma=0.2, size=candidate.shape)
        mutated = candidate * (1 + self.params.mutation_rate * (noise - 1))
        mutated = np.clip(mutated, 1e-6, None)
        return mutated

    def optimise(self, seed_weights: np.ndarray, fitness_fn: FitnessFunction) -> np.ndarray:
        seed_weights = np.asarray(seed_weights, dtype=float)
        seed_weights = seed_weights / seed_weights.sum()

        population = self._initialise_population(seed_weights)
        best_candidate = population[0]
        best_score = fitness_fn(best_candidate)

        for _ in range(self.params.generations):
            scores = np.array([fitness_fn(individual) for individual in population])
            gen_best_idx = np.argmin(scores)
            if scores[gen_best_idx] < best_score:
                best_candidate = population[gen_best_idx]
                best_score = scores[gen_best_idx]

            new_population = []
            while len(new_population) < self.params.population_size:
                parent_a = self._tournament_select(population, scores)
                parent_b = self._tournament_select(population, scores)
                child = parent_a.copy()
                if self.rng.random() < self.params.crossover_rate:
                    child = self._crossover(parent_a, parent_b)
                if self.rng.random() < self.params.mutation_rate:
                    child = self._mutate(child)
                new_population.append(child)

            population = self._normalise(np.array(new_population))

        best_candidate = best_candidate / best_candidate.sum()
        return best_candidate


