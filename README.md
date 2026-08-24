# Movie Recommendation Kaggle Hackathon
### Part of the ALX Movie Recommendation Kaggle Hackathon 2026

## 📌 Project Overview
This project is a machine learning solution developed to predict user movie ratings on a scale of 0.5 to 5.0. By leveraging historical interaction data, the system aims to provide personalized recommendations that reduce "choice overload" for users.

**The Primary Goal:** Minimize the Root Mean Squared Error (RMSE) of movie rating predictions.

## 🏗 Repository Structure
- `data/`: Contains raw competition data (ignored by Git) and processed features.
- `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and prototyping.
- `src/`: Modular Python source code for data ingestion, cleaning, and model architecture.
- `models/`: Serialized trained model binaries (.pkl).
- `reports/`: Final performance analysis and visualizations.

## 🛠 Technology Stack
- **Languages:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Virtualenv / Git Bash

## 🚀 Methodology
1. **Exploratory Data Analysis (EDA):** Analyzing matrix sparsity, user-rating distributions, and item popularity.
2. **Feature Engineering:** Utilizing movie metadata (genres, release year) and user interaction history.
3. **Modeling:** Implementation of Collaborative Filtering (SVD/ALS) and Content-Based Filtering.
4. **Validation:** K-Fold Cross-Validation to ensure robustness against the RMSE metric.

## 📝 Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/berof3/Movie-Recommendation-Kaggle-Hackathon.git
2. Create Virtual Enviroment and Install dependencies

    ```bash
    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt

Note: This project is part of a private Kaggle Hackathon. Data files are not included in this repository in compliance with competition rules.

