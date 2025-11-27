# SMOTE + ENN Oversampler
import pandas as pd
from imblearn.combine import SMOTEENN
from typing import Callable, Tuple
from config import RANDOM_SEED

def create_smote_enn_sampler() -> Callable[[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    def _sampler(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        sampler = SMOTEENN(random_state=RANDOM_SEED)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    return _sampler