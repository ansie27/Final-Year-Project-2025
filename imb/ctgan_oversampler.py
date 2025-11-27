# CTGAN Oversampler
import pandas as pd
from typing import Callable, List, Tuple
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

def create_ctgan_sampler(
    feature_columns: List[str],
    discrete_columns: List[str],
    target_column: str,
    epochs: int,
) -> Callable[[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:

    def _sampler(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        train_df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_df)

        for col in set(discrete_columns + [target_column]):
            if col in train_df.columns:
                metadata.update_column(column_name=col, sdtype="categorical")

        model = CTGANSynthesizer(
            metadata=metadata,
            epochs=epochs,
            verbose=False,
            batch_size=min(512, len(train_df)),
        )
        model.fit(train_df)

        counts = y.value_counts()
        max_count = counts.max()
        synthetic_parts = []

        for label, count in counts.items():
            deficit = int(max_count - count)
            if deficit <= 0:
                continue
            condition_df = pd.DataFrame({target_column: [label] * deficit})
            try:
                synthetic = model.sample_from_conditions(conditions=condition_df)
            except Exception:
                synthetic = model.sample(deficit)
                synthetic = synthetic[synthetic[target_column] == label]
                if synthetic.empty:
                    continue
                synthetic = synthetic.head(deficit)
            synthetic_parts.append(synthetic)

        if synthetic_parts:
            synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
            augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
        else:
            augmented_df = train_df

        X_aug = augmented_df[feature_columns].copy()
        y_aug = augmented_df[target_column].copy()

        for col in X_aug.columns:
            X_aug[col] = pd.to_numeric(X_aug[col], errors="coerce")

        return X_aug, y_aug

    return _sampler