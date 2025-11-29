import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from mcdm.fuzzy_ahp import run_fuzzy_ahp_analysis
from mcdm.fuzzy_evaluation import FuzzyPipelineEvaluator
from mcdm.fuzzy_topsis import run_fuzzy_topsis
from src.utils import ensure_directory, print_progress, print_section_header  # type: ignore[import]

PIPELINE_REPORT_PATH = PROJECT_ROOT / "outputs" / "fuzzy_ahp_topsis_results.json"

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


def main() -> None:
    print_section_header("Running Fuzzy AHP → Fuzzy TOPSIS pipeline")
    ahp_output = run_fuzzy_ahp_analysis()
    topsis_output = run_fuzzy_topsis(weights_data=ahp_output["weights_payload"])

    evaluator = FuzzyPipelineEvaluator(
        standard_weights=ahp_output["standard_weights"],
        ga_weights=ahp_output["ga_weights"],
        standard_scores=topsis_output["standard_scores"],
        ga_scores=topsis_output["ga_scores"],
        suppliers=topsis_output["suppliers"],
        commodities=topsis_output["commodities"],
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