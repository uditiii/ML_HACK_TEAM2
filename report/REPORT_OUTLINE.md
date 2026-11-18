# PathSnap — Team 2 Report (skeleton)

> Follows `Deliverables.pdf` section by section.
> Every `[[...]]` placeholder must be replaced with **real numbers from our own runs**. Each placeholder names its owner.
> Every figure and number must be reproducible from the GitHub repo (give the script or notebook path next to it).
> State the **dataset partition** for every analysis (train / student_test).

---

## 1. Team and repository
- Team members and roles: Person A [[name]], Person B [[name]], Person C [[name]], Person D [[name]]
- GitHub repository: [[URL]] — commit used for this report: [[hash]]
- How to reproduce: [[README section / commands]]

---

## 2.2 Dataset Understanding and Exploratory Data Analysis
**Owner: Person A.** Source: `notebooks/01_preprocessing.ipynb`, `notebooks/01b_eda_plots.ipynb`, `results/preprocessing_log.md`

### 2.2.1 Dataset inventory
| Partition | Trips | GPS points | Points per trip (min / median / max) |
|---|---|---|---|
| train | 80 | [[from preprocessing_report.csv]] | [[ ]] |
| student_test | 10 | [[ ]] | [[ ]] |
| evaluator_test | 10 | not accessed | — |

### 2.2.2 Q1 — How did you preprocess the data?
[[Person A: summarise steps a–h of `clean_track` and paste the table from `results/preprocessing_log.md`, re-run with GAP_THRESHOLD_S = 30]]
- Missing-value analysis: [[dropped_invalid totals per partition]]
- Duplicates / ordering / duplicate timestamps: [[counts]]
- Gap handling: segments split at dt > 30 s — [[n gaps, n trips]]
- Projection: local equirectangular in metres; justification [[ ]]
- Leakage controls: cleaning uses the track only; evaluator data never loaded

### 2.2.3 Q2 — Patterns observed; features extracted and why
- Patterns: [[e.g. ~96.5 % of steps are exactly 1 s (train); trips span several continents; network size varies widely]]
- Features (owner: [[feature owner]]):

| Feature | Definition | Why it helps |
|---|---|---|
| [[distance to candidate arc]] | [[ ]] | [[ ]] |
| [[heading difference]] | [[ ]] | [[ ]] |
| [[ ... ]] | | |

### 2.2.4 Required figures (each caption names the partition)
- Fig. 2.1 Points per trip — `results/figures/points_per_trip_train.png` (train)
- Fig. 2.2 dt histogram, log scale — `results/figures/dt_hist_train.png` (train)
- Fig. 2.3 Largest gap per trip — `results/figures/max_gap_per_trip_train.png` (train)
- Fig. 2.4 Trip centres — `results/figures/trip_centres_train.png` + interactive `trip_centres_train.html` (train)
- Fig. 2.5 Road network size per trip — `results/figures/network_size_train.png` (train)
- Fig. 2.6 GPS trajectory + road network (Folium) — `results/maps/trip_<id>.html` screenshot ([[partition/trip]])

---

## 3. Model Experimentation and Comparison
Minimum five models; document **every** model tried.

### 3.1 Overall model comparison
**Owner: Person D** (fills from experiment logs). If a metric is not applicable, write "N/A" and give the reason in the cell.

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route continuity | Student-Test Macro F1 | Log Loss | Inference Time |
|---|---|---|---|---|---|---|---|---|
| Gaussian Naive Bayes | Probabilistic / generative | [[var_smoothing=…]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |
| [[Model 2]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |
| [[Model 3]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |
| [[Model 4]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |
| [[Model 5]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |

### 3.2 Per-model subsections
Copy this block once per model. Answers must refer to **our** results, not textbook descriptions.

#### 3.2.x [[Model name]] — owner [[name]]
**A. Model understanding**
1. Why is this algorithm suitable for PathSnap? [[ ]]
2. Major strengths and limitations for this task? [[ ]]

**B. Training and configuration**
5. Important hyperparameters and their initial values: [[ ]]
6. Loss function / optimisation objective: [[ ]]

**C. Experimental results**
7. Performance on student_test: [[ ]]
8. Observation-level Top-1 Accuracy / Macro F1 / Log Loss on student_test: [[ ]] / [[ ]] / [[ ]]
9. Overfitting or underfitting? Evidence (train vs CV vs test gap, curves if any): [[ ]]

### 3.3 Model visualisations
**Owner: Person D.** Every graph from actual results; no artificial training curves for models without epoch histories.
- [[CV Macro F1 comparison across all models]] — `results/figures/[[ ]]`
- [[Student-test Top-1 and Macro F1 comparison]] — `results/figures/[[ ]]`
- [[Log Loss comparison (where applicable)]] — `results/figures/[[ ]]`
- [[Confusion matrices (optional)]]
- [[Train/validation loss curves — MLP only]]
- [[Plot justifying final model selection]]

---

## 4. Bayesian Hyperparameter Optimisation (chosen model)
**Owner: [[name]]**
1. Hyperparameters searched: [[ ]]
2. Ranges / distributions: [[ ]]
3. Objective optimised: [[e.g. 5-fold CV Macro F1]]
4. Number of trials: [[ ]]
5. Best values: [[ ]]
6. Did tuning beat the untuned model? [[ ]]

| Model | CV Macro F1 | Student-Test Top-1 | Student-Test Macro F1 | Log Loss | Inference Time |
|---|---|---|---|---|---|
| Untuned [[model]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |
| Tuned [[model]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |

Same metrics and same partitions for both rows. Optimisation trial log: `[[path]]`.

---

## 5. HMM and Viterbi Route Reconstruction
**Owner: [[name]]**
1. Hidden states: [[ ]]
2. Emission probabilities: [[ ]]
3. Transition probabilities: [[ ]]
4. Missing candidates / large GPS gaps (segment_id from preprocessing, threshold 30 s): [[ ]]
5. Consecutive observations suggesting disconnected road segments: [[ ]]
6. Did HMM decoding improve the route? [[evidence]]

| Method | Top-1 Accuracy (student_test) | Macro F1 | Route continuity |
|---|---|---|---|
| Classifier only | [[ ]] | [[ ]] | [[ ]] |
| Classifier + HMM/Viterbi | [[ ]] | [[ ]] | [[ ]] |

Cases where HMM did **not** help: [[trip ids + reason]]

---

## 6. Geographical Visualisation Using Folium
**Owner: Person A.** Code: `src/visualisation/maps.py`, `scripts/make_report_maps.py`.
Layers: raw GPS (blue points), road network near the trip (grey), reference route (green), predicted route (red dashed); layer control, legend, title, zoom fitted to the GPS extent.

| Map | Trip (student_test) | Why chosen | File |
|---|---|---|---|
| Short trip | [[00000052]] | fewest points | `results/maps/report_[[id]].html` |
| Long trip | [[00000056]] | most points | `results/maps/report_[[id]].html` |
| Big gap | [[00000065]] | largest gap | `results/maps/report_[[id]].html` |

[[Screenshot of at least one map + 2–3 sentences: where predicted and reference agree or differ]]

---

## 7. API, UI and Deployment
**Owners: API [[name]] (Laptop 1), UI Person A (Laptop 2)**
- Architecture: Streamlit UI (Laptop 2) → HTTP → FastAPI (Laptop 1). The UI sends only `trip_id`; ground truth stays on the server.
- Run commands: API [[ ]]; UI `API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py`
- [[Screenshot: UI with trip selected, metrics and map]]
- [[Screenshot: successful API request/response in the CMD of the receiving laptop]]
- The demonstrated code matches commit [[hash]].

---

## 8. AI-assistance disclosure
| Part | Tool | What it was used for | What the team did / verified |
|---|---|---|---|
| [[e.g. loader / preprocessing / maps / UI scaffolding]] | [[tool name]] | [[ ]] | [[ran, tested, understood and can explain the code; all numbers come from our own runs]] |
| [[ ]] | | | |

Statement: [[All results in this report were produced by running our own code on the provided data. AI tools did not replace experimentation or analysis.]]
