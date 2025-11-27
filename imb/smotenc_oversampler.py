import pandas as pd
from imblearn.over_sampling import SMOTENC
from typing import Callable, List, Tuple
from config import RANDOM_SEED


def create_smotenc_sampler(
    categorical_features: List[int],
) -> Callable[[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """Return a sampler function that applies SMOTENC to the given dataset."""

    def _sampler(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        sampler = SMOTENC(
            categorical_features=categorical_features,
            sampling_strategy="auto",
            random_state=RANDOM_SEED,
        )
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    return _sampler

