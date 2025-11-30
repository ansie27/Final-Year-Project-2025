import json
import sys
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from mcdm.fuzzy_ahp import run_fuzzy_ahp_analysis
from mcdm.fuzzy_evaluation import FuzzyPipelineEvaluator
from mcdm.fuzzy_topsis import run_fuzzy_topsis

PIPELINE_REPORT_PATH = PROJECT_ROOT / "outputs" / "fuzzy_ahp_topsis_results.json"
DEFAULT_DATASET_PATH = Path(config.ENGINEERED_DATA_PATH)


def _gpu_is_available() -> bool:
    return torch is not None and torch.cuda.is_available()

def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def print_section_header(title: str) -> None:
    border = "=" * len(title)
    print(f"\n{border}\n{title}\n{border}")

def print_progress(message: str) -> None:
    print(f"[progress] {message}")

def persist_pipeline_summary(summary: dict) -> None:
    ensure_directory(PIPELINE_REPORT_PATH.parent)
    if PIPELINE_REPORT_PATH.exists():
        with PIPELINE_REPORT_PATH.open("r", encoding="utf-8") as handle:
            try:
                existing = json.load(handle)
            except json.JSONDecodeError:
                existing = {}
    else:
        existing = {}
    existing["mcdm_pipeline"] = summary
    with PIPELINE_REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
    print_progress(f"Pipeline summary written to {PIPELINE_REPORT_PATH}")

def main(dataset_path: Path | str | None = None) -> None:
    resolved_dataset = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    print_section_header("Running Fuzzy AHP-TOPSIS pipeline")
    print_progress(f"Using dataset at {resolved_dataset}")
    ahp_output = run_fuzzy_ahp_analysis(dataset_path=resolved_dataset)
    prefer_gpu = _gpu_is_available()
    if prefer_gpu:
        print_progress("GPU detected - enabling accelerated TOPSIS computations")
    else:
        print_progress("GPU not detected or unavailable; running TOPSIS on CPU")
    weights_payload = ahp_output["weights_payload"]
    topsis_standard = run_fuzzy_topsis(
        weights_data=weights_payload,
        use_gpu=prefer_gpu,
        weight_key="standard",
    )

    if "ga_optimised" in weights_payload:
        topsis_ga = run_fuzzy_topsis(
            weights_data=weights_payload,
            use_gpu=prefer_gpu,
            weight_key="ga_optimised",
        )
    else:
        print_progress("GA weights not available; reusing standard TOPSIS results")
        topsis_ga = topsis_standard

    evaluator = FuzzyPipelineEvaluator(
        standard_weights=ahp_output["standard_weights"],
        ga_weights=ahp_output["ga_weights"],
        standard_scores=topsis_standard["scores"],
        ga_scores=topsis_ga["scores"],
        suppliers=topsis_standard["suppliers"],
        commodities=topsis_standard["commodities"],
    )
    summary = evaluator.generate_summary()

    report = {
        "weight_delta_max": summary.weight_delta_max,
        "weight_delta_mean": summary.weight_delta_mean,
        "ranking_correlation": summary.ranking_correlation,
        "score_shift_mean": summary.score_shift_mean,
        "top_changes": summary.top_changes,
        "consistency": ahp_output["consistency"],
    }

    print_progress(
        f"Ranking correlation between standard and GA pipelines: {summary.ranking_correlation:.4f}"
    )
    if summary.top_changes:
        print_progress(f"Differing top entries: {', '.join(summary.top_changes)}")
    else:
        print_progress("No differences detected in top-ranked supplier-commodity pairs")

    persist_pipeline_summary(report)

if __name__ == "__main__":
    main()