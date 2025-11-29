import subprocess
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXECUTABLE = sys.executable or "python"

MODEL_SCRIPTS = {
    "ANN": PROJECT_ROOT / "ml_models" / "ann_model.py",
    "Random Forest": PROJECT_ROOT / "ml_models" / "random_forest_model.py",
    "XGBoost": PROJECT_ROOT / "ml_models" / "xgboost_model.py",
}

RESULT_FILES = {
    "ANN": PROJECT_ROOT / "outputs" / "models" / "ann_results.yaml",
    "Random Forest": PROJECT_ROOT / "outputs" / "models" / "random_forest_results.yaml",
    "XGBoost": PROJECT_ROOT / "outputs" / "models" / "xgboost_results.yaml",
}

EVALUATION_SCRIPT = PROJECT_ROOT / "ml_models" / "ml_evaluation.py"

def run_command(command: list[str]) -> None:
    process = subprocess.run(command, check=True)

def train_model(model_name: str, script_path: Path) -> None:
    command = [PYTHON_EXECUTABLE, str(script_path)]
    print(f"\n[TRAIN] Starting {model_name} training.")
    run_command(command)
    print(f"[TRAIN] {model_name} training complete.\n")

def evaluate_models() -> str:
    command = [PYTHON_EXECUTABLE, str(EVALUATION_SCRIPT)]
    print("\n[EVAL] Comparing models.")
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    print(process.stdout)

    summary_line = next(
        (line for line in process.stdout.splitlines() if line.lower().startswith("overall best")), ""
    )
    best_model = summary_line.split(":")[-1].strip() if summary_line else "Unknown"
    return best_model

def main() -> None:
    for name, script in MODEL_SCRIPTS.items():
        train_model(name, script)

    best_model = evaluate_models()
    print(f"\n[RESULT] Based on current metrics, the recommended model is: {best_model}")

if __name__ == "__main__":
    main()