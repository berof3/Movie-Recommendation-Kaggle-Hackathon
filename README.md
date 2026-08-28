# Movie Recommendation Kaggle Hackathon
### Part of the ALX Movie Recommendation Kaggle Hackathon 2026

## Project Overview
This project is a machine learning solution developed to predict user movie ratings on a scale of 0.5 to 5.0. By leveraging historical interaction data, the system aims to provide personalized recommendations that reduce "choice overload" for users.

**The Primary Goal:** Minimize the Root Mean Squared Error (RMSE) of movie rating predictions.

## Repository Structure
- `data/`: Contains raw competition data (ignored by Git) and processed features.
- `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and prototyping.
- `src/`: Modular Python source code for data ingestion, cleaning, and model architecture.
- `models/`: Serialized trained model binaries (.pkl).
- `reports/`: Final performance analysis and visualizations.

## Technology Stack
- **Languages:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Virtualenv / Git Bash

## Methodology
1. **Exploratory Data Analysis (EDA):** Analyzing matrix sparsity, user-rating distributions, and item popularity.
2. **Feature Engineering:** Utilizing movie metadata (genres, release year) and user interaction history.
3. **Modeling:** Custom matrix factorization (Funk-SVD-style collaborative filtering) and Content-Based Filtering, combined into a **hybrid blend** weighted by each movie's training-rating count.
4. **Validation:** 5-fold Cross-Validation to ensure robustness against the RMSE metric, plus a dedicated item cold-start holdout.

## Results

The pipeline is complete end to end - see [`reports/final_report.md`](reports/final_report.md) for the full write-up (including bugs hit and fixed along the way) and `notebooks/02_results_report.ipynb` for how the figures below were generated.

**5-fold CV, model comparison** (mean RMSE ± std; lower is better):

| Model | CV mean RMSE | std |
|---|---|---|
| Global mean | 1.0611 | 0.0006 |
| Bias baseline | 0.8654 | 0.0005 |
| Content-based | 0.9245 | 0.0006 |
| Matrix factorization | 0.8663 | 0.0022 |

![Model comparison, 5-fold CV](figures/model_comparison_cv.png)

**Final hybrid system**, weighted to match `test.csv`'s real composition (87.3% regular rows / 12.7% item cold-start rows - movies never seen in `train.csv`):

| Scenario | MF-only system | Hybrid system |
|---|---|---|
| Regular rows | 0.8540 | 0.8495 |
| Cold-start rows | 0.9816 | 0.9275 |
| **Overall** | **0.8713** | **0.8598** |

![Hybrid blending impact](figures/hybrid_impact.png)

Blending in content-based predictions - weighted by how many training ratings each movie had - improves RMSE in every scenario, most sharply on item cold start (~5.5% relative), for an overall **~1.3% relative RMSE improvement** over matrix factorization alone. The final Kaggle submission (`submission.csv`, gitignored - regenerate via `src/evaluation.py`'s `generate_submission()`) is produced by this hybrid model trained on all of `train.csv`.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/berof3/Movie-Recommendation-Kaggle-Hackathon.git
2. Create Virtual Enviroment and Install dependencies

    ```bash
    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt

Note: This project is part of a private Kaggle Hackathon. Data files are not included in this repository in compliance with competition rules.

