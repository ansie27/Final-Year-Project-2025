# ADASYN Oversampler
import pandas as pd
from imblearn.over_sampling import ADASYN
from typing import Callable, Tuple
from config import RANDOM_SEED


def create_adasyn_sampler() -> Callable[[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """Return a sampler function that applies ADASYN to the given dataset."""

    def _sampler(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        sampler = ADASYN(random_state=RANDOM_SEED)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    return _sampler

