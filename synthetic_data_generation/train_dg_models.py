import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence, Type

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

# Ensure project root is importable for config reuse
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_PATH

OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"
ROW_TARGETS: Sequence[int] = (500, 1000, 5000, 10000, 50000)
ORDINAL_INT_COLUMNS: Sequence[str] = (
    "Supplier_Tier",
    "Compliance_Level",
    "Sustainability_Report_Availability",
    "Incident_History_Count",
)
EXCLUDED_COLUMNS: Sequence[str] = (
    "Supplier_ID",
    "Supplier_Name",
    "Commodity_ID",
    "Commodity_Name",
)

MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "CTGAN": {
        "class": CTGANSynthesizer,
        "kwargs": {
            "epochs": 300,
            "verbose": True,
        },
    },
    "TVAE": {
        "class": TVAESynthesizer,
        "kwargs": {
            "epochs": 300,
            "verbose": True,
        },
    },
}


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load the preprocessed dataset and normalise dtypes for SDV models."""
    df = pd.read_csv(path)
    df = df.convert_dtypes()

    existing_exclusions = [col for col in EXCLUDED_COLUMNS if col in df.columns]
    if existing_exclusions:
        df = df.drop(columns=existing_exclusions)

    # Convert nullable integers to standard ints (CTGAN/TVAE require numpy dtypes)
    int_cols = df.select_dtypes(include=["Int64"]).columns
    for col in int_cols:
        df[col] = df[col].astype("int64")
    return df


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """Create SDV metadata for the given dataframe."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    for col in ORDINAL_INT_COLUMNS:
        if col in df.columns:
            metadata.update_column(column_name=col, sdtype="numerical")
    return metadata

def train_and_generate(
    model_name: str,
    model_class: Type[CTGANSynthesizer] | Type[TVAESynthesizer],
    df: pd.DataFrame,
    row_targets: Sequence[int],
) -> None:
    """Fit the model and export synthetic samples for all requested sizes."""
    metadata = build_metadata(df)
    model_kwargs = MODEL_REGISTRY[model_name]["kwargs"]
    print(f"[{model_name}] Training on {len(df):,} rows...")

    synthesizer = model_class(metadata, **model_kwargs)
    synthesizer.fit(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rows in row_targets:
        print(f"[{model_name}] Generating {rows:,} synthetic rows...")
        synthetic_df = synthesizer.sample(num_rows=rows)
        output_path = OUTPUT_DIR / f"{model_name}_{rows}_data.csv"
        synthetic_df.to_csv(output_path, index=False)
        print(f"    Saved -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CTGAN and TVAE models on the processed dataset and "
        "generate multiple synthetic samples.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROCESSED_DATA_PATH,
        help="Path to the processed CSV file (default: config.PROCESSED_DATA_PATH)",
    )
    parser.add_argument(
        "--row-targets",
        type=int,
        nargs="+",
        default=list(ROW_TARGETS),
        help="List of sample sizes to generate for each model.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Subset of models to train (default: both).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    df = load_processed_data(args.data_path)
    for model_name in args.models:
        registry_entry = MODEL_REGISTRY[model_name]
        train_and_generate(model_name, registry_entry["class"], df, args.row_targets)

    print("Synthetic dataset generation complete.")


if __name__ == "__main__":
    main()
