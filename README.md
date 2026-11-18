# PathSnap — Map-Matching & Route Reconstruction (Team 2)

Noisy GPS points → candidate road arcs (≤50 m) → Random Forest scores each candidate → HMM + Viterbi reconstructs a connected route → FastAPI (Laptop 1) + Streamlit UI (Laptop 2).

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put the dataset (not committed) at `data/raw/Student folder/{train,student_test}/` and the evaluator trips at `data/raw/evaluator_test/`.

## Reproduce
```bash
python scripts/build_candidates.py --partitions train student_test   # labels (v2) + candidates
python scripts/build_tables.py                                        # features v2
python scripts/run_models_v2.py                                       # 5-model comparison -> results/model_results_v2.csv
python scripts/tune_rf.py v2                                          # Bayesian optimisation (Optuna) -> artifacts/models/final_model.joblib
python scripts/run_hmm_experiment.py                                  # classifier-only vs HMM -> results/hmm_comparison.csv
```

## Run the demo
Laptop 1 (API):
```bash
EVAL_DIR="data/raw/evaluator_test" uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Laptop 2 (UI):
```bash
API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
```

## Repository layout
`src/data` loading/cleaning · `src/candidates` labels + candidates · `src/features` features · `src/models` metrics, CV framework, models · `src/hmm` Viterbi · `src/pipeline` end-to-end prediction · `api/` FastAPI · `ui/` Streamlit · `scripts/` runnable steps · `notebooks/` EDA, feature selection, models, HMM, plots · `results/` CSVs, figures, maps · `artifacts/models` final model + metadata

## AI assistance
AI tools were used to assist with code; all experiments were run by the team and all reported numbers come from this repository's outputs.