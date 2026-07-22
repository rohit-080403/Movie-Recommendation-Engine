# Movie Recommendation & Success Prediction System

## Overview
This project combines a **content-based movie recommender system** with a full **machine learning pipeline** to predict whether a movie will be commercially successful (a "hit"), using the TMDB 5000 Movies dataset. It demonstrates data analysis, multiple supervised learning algorithms, cross-validation, boosting, unsupervised clustering, and dimensionality reduction on a single dataset.

## Dataset
`movies.csv` — 4803 movies, 24 columns including budget, revenue, genres, keywords, cast, crew, director, popularity, vote_average, vote_count, runtime, and release_date.

## Project Structure (Phases)

| Phase | Description |
|---|---|
| 1 | Data loading & inspection (NumPy, Pandas) |
| 2 | Exploratory Data Analysis (Matplotlib, Seaborn) |
| 3 | Content-based recommender using TF-IDF + Cosine Similarity |
| 4 | Feature engineering + Supervised learning (Logistic Regression, Decision Tree, Random Forest) |
| 5 | Cross-validation (5-fold Stratified K-Fold) |
| 6 | Boosting algorithms (AdaBoost, XGBoost) |
| 7 | Unsupervised learning (KMeans clustering) |
| 8 | Dimensionality reduction (PCA & LDA) |

## Target Variable
`is_hit` (binary): 1 if a movie's revenue exceeded its budget, 0 otherwise. Built from the subset of movies with reported budget and revenue (~3200 rows).

## Features Used
- Numeric: `budget`, `popularity`, `runtime`, `vote_average`, `vote_count`, `genre_count`, `release_year`
- One-hot encoded top 10 genres

## Results

| Metric | Value |
|---|---|
| Best Model | Random Forest (n_estimators=100, max_depth=None) |
| Test Accuracy | 81.9% |
| F1 Score | 0.883 |
| Cross-Validation Accuracy | 81.3% ± 2.0% |
| ROC-AUC | 0.84 |
| Top Predictive Features | vote_count, genre_Thriller, release_year, genre_Action |
| PCA Variance Captured (2 components) | 34.6% |

## Key Insights
- Vote ratings cluster around 5.5–7/10; budget and revenue show a positive but noisy relationship.
- Audience engagement (`vote_count`) and genre (Thriller, Action) are stronger predictors of commercial success than budget alone.
- Random Forest and boosting models outperform linear models, indicating non-linear feature interactions drive success.
- PCA/LDA comparison shows the difference between unsupervised (variance-based) and supervised (class-separation-based) dimensionality reduction.

## Tech Stack
- Python, NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Decision Tree, Random Forest, KMeans, PCA, LDA, cross-validation)
- XGBoost, AdaBoost
- Pickle (model persistence)

## How to Run
1. Install dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn xgboost`
2. Place `movies.csv` in the project directory
3. Run the notebook cells in order (Phase 1 → Phase 8)
4. Trained model is saved as `best_rf_model.pkl`, scaler as `scaler.pkl`

## Loading the Saved Model
```python
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

