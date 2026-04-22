# AI Jobs — Data Cleaning and Analysis (2026)

A student project demonstrating data cleaning, NLP feature engineering, and exploratory analysis on a consolidated global AI jobs dataset. The repository contains a reproducible Jupyter Notebook pipeline that loads raw data, performs cleaning and salary imputation, applies an NLP pipeline and TF‑IDF feature extraction, and produces visualizations used for interpretation.

## Table of Contents
- Project Overview
- Dataset & Licensing
- Preprocessing Summary
- Analysis & Key Findings
- Repository Structure
- Reproducibility
- Limitations
- License & Contact

## Project Overview

This project cleans and analyzes job posting data related to artificial intelligence roles. The primary goals are:
- demonstrate a reproducible data cleaning workflow;
- build a simple NLP pipeline for job description text (tokenization, stopwords removal, lemmatization);
- extract TF‑IDF features and compute correlations with salary;
- produce clear visualizations suitable for a portfolio presentation.

## Dataset & Licensing

- Source: original consolidated CSV (internal dataset used for the assignment). If the original raw dataset cannot be published, a cleaned sample or instructions are provided to reproduce the dataset locally.
- Cleaned dataset produced by the pipeline: `output/ai_jobs_global_2026_cleaned.csv`.

Before publishing this repository publicly, confirm you have the right to share the raw data. If not, remove raw files and keep only the derived `output/` artifacts or provide a data acquisition script.

## Preprocessing Summary

- Convert salary fields to numeric and create a unified `salary` column (row-wise mean of `salary_min`/`salary_max` when available).
- Impute missing `salary` values with the dataset median, then backfill `salary_min`/`salary_max` from `salary` so no `salary_min`/`salary_max` remain missing.
- Text preprocessing pipeline applied to job descriptions:
  - lowercase, strip HTML/punctuation
  - tokenization with regex
  - stopword removal (NLTK stopwords)
  - POS-aware lemmatization (NLTK WordNetLemmatizer)
- TF‑IDF vectorization (scikit-learn's `TfidfVectorizer`) used for feature extraction; correlations between TF‑IDF term scores and salary were computed to identify skill/term relationships.

## Analysis & Key Findings

- The cleaned dataset contains 5,773 rows after preprocessing.
- Salary fields were fully imputed (no remaining missing `salary_min`/`salary_max`).
- Top correlated terms with salary and visualizations (heatmap + bar chart) are saved for interpretation in `output/`.

## Repository Structure

- `load_clean_ai_jobs_2026.ipynb` — main notebook (Load → Clean → Chapter 3 → Chapter 4 analysis).
- `output/ai_jobs_global_2026_cleaned.csv` — cleaned dataset produced by notebook.
- `output/` — contains generated figures: wordcloud, n‑grams bar chart, salary boxplot, TF‑IDF heatmap and top correlation chart.
- `requirements.txt` — exact Python package versions used for reproducibility.

## Reproducibility

Recommended steps to reproduce the results locally (Windows / PowerShell shown):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Run the notebook end-to-end (this will execute all cells and recreate outputs)
jupyter nbconvert --to notebook --execute load_clean_ai_jobs_2026.ipynb --ExecutePreprocessor.timeout=600
```

If you prefer to run interactively:

```bash
pip install -r requirements.txt
jupyter lab
# open load_clean_ai_jobs_2026.ipynb and use Kernel → Restart & Run All
```

If the raw dataset is not included here for licensing reasons, run the provided data acquisition/preprocessing steps in the notebook to recreate `output/ai_jobs_global_2026_cleaned.csv`.

## Limitations

- The salary imputation strategy uses the dataset median for missing `salary` values and then backfills `salary_min`/`salary_max`; other imputation strategies (global mean, model-based) may be more appropriate depending on use case.
- The TF‑IDF correlation analysis is univariate (term vs salary) and does not control for confounders (location, experience, company type).
- This repository is intended for educational and portfolio use; do not treat results as production-grade insights without further validation.

## License & Contact

This project is provided for educational use. Add a license (for example, `LICENSE` with MIT) if you plan to publish publicly.

For questions or collaboration, contact: andyderis36@gmail.com

---

If you’d like, I can also:
- add a `LICENSE` file (MIT),
- generate a polished short summary for the GitHub project description, or
- strip outputs from the notebook and clean metadata before commit.
