# Mapping Technical Skills in AI/ML Job Markets: Analysis, Modeling, and Interpretability

This repository contains the complete end-to-end Python implementation for **STINK3024 Machine Learning (A252)** assignments. The project maps technical skill mentions in unstructured global AI/ML job postings (5,773 records) and implements a high-fidelity machine learning modeling pipeline to predict annual base salary brackets.

---

## 📂 Repository Structure

The project is structured into three phases, each represented by a dedicated Jupyter Notebook and associated output artifacts:

```
├── (copy)descriptive_text_analytics.ipynb  # Backup of Phase 1 NLP notebook
├── (preview)descriptive_text_analytics.html # Phase 1 HTML export
├── descriptive_text_analytics.ipynb        # Phase 1 Notebook: Ingestion, Text Cleaning & EDA
├── (a2)machine_learning_modelling.ipynb        # Phase 2 Notebook: Preliminary Model Development
├── (a3)(a2)machine_learning_modelling.ipynb    # Phase 3 Notebook: Final Modeling, Diagnostics & SHAP
│
├── A1/                                      # Phase 1 Documentation & Report Draft
│   └── a1_code_documentation.md             # Code Report for Phase 1 (Full English)
├── A2/                                      # Phase 2 Rubric & Report Draft
│   └── a2_code_documentation.md             # Code Report for Phase 2 (Full English)
├── A3/                                      # Phase 3 Rubric & Complete Final Report Draft
│   ├── A252_Project Machine Learning Development and Reporting (20%).pdf
│   ├── a3_code_documentation.md             # Code Report for Phase 3 (Full English)
│   └── a3_report_draft_and_guidelines.md    # Complete 40-50 Page Report Draft (Pure English)
│
├── output/                                  # Phase 1 output csv and EDA plots
│   ├── ai_jobs_global_2026_cleaned.csv     # Preprocessed dataset (5,773 postings)
│   ├── ngrams_ch4.png
│   ├── salary_boxplot_ch4.png
│   ├── tfidf_salary_heatmap_ch4.png
│   └── wordcloud_ch4.png
│
├── output_modelling/                        # Phase 2 & 3 evaluation results & model diagnostics
│   ├── cleaned_tfidf_top_skills.png        # Filtered technical vocabulary barplot
│   ├── model_evaluation_metrics.csv        # Metrics summary (RMSE, MAE, R2)
│   ├── model_performance_comparison.png    # RMSE and R2 bar charts comparison
│   ├── residual_plots.png                  # Multi-model residual homoscedasticity scatterplots
│   ├── learning_curve_rf.png               # Training vs Validation RMSE learning curve
│   ├── rf_feature_importances.png          # Gini Feature Importances (Top 20)
│   └── shap_summary_plot.png               # SHAP value impact beeswarm plot
│
├── requirements.txt                         # Python package dependency manifest
└── README.md                                # Repository index documentation
```

---

## ⚡ Technical Stack & Requirements
*   **Programming Language:** Python 3.11+
*   **NLP Processing:** `nltk` (WordNet lemmatizer, tokenizers, POS tagging), `re`
*   **Data Wrangling:** `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn` (Ridge, RandomForestRegressor, MLPRegressor, Grid & Randomized CV wrappers)
*   **Hardware Acceleration:** `scikit-learn-intelex` (Intel(R) Scikit-Learn MKL Extension)
*   **Interpretability:** `shap` (SHapley Additive exPlanations)
*   **Data Visualization:** `matplotlib`, `seaborn`, `wordcloud`

---

## 🔬 Phase 1: Descriptive Text Analytics & Preprocessing
*   **Notebook:** [descriptive_text_analytics.ipynb](descriptive_text_analytics.ipynb)
*   **Pipeline & Mechanics:**
    1.  **Ingestion:** Ingests the Kaggle "AI Job Market — Global 2026" dataset. Normalizes continuous salary fields (`salary_min`/`salary_max`), imputing missing targets using the corpus median salary.
    2.  **Linguistic Cleaning:** Strips HTML formatting, punctuation, and digits. Tokenizes text using a regular expression tokenizer and filters out standard NLTK stopwords.
    3.  **POS-Aware Lemmatization:** Uses the NLTK WordNet Lemmatizer conjoined with part-of-speech tags to convert nouns, verbs, adjectives, and adverbs into their standard base forms.
    4.  **EDA & Visuals:** Generates word clouds, salary distribution boxplots across experience levels, and extracts top correlated terms utilizing TF-IDF.

---

## 🤖 Phase 2 & 3: Machine Learning Modeling, Diagnostics & Interpretability
*   **Notebook:** [(a3)(a2)machine_learning_modelling.ipynb]((a3)(a2)machine_learning_modelling.ipynb)
*   **Pipeline & Mechanics:**

```
                  RAW CLEANED DATASET (5,773 rows)
                                 │
                   [ 70:30 Split; random_state=42 ]
                                 ▼
                     TRAINING SET & TESTING SET
                                 │
              [ Custom Blacklist Stopwords Filter ]
               (Wipes out HTML residues, German
              leakage, and domain-specific noise)
                                 ▼
                  Retained Skills & Context Words
                                 │
                [ Vectorizer: TF-IDF (1,000 Dimensions) ]
                (fit_transform on Train | transform on Test)
                                 ▼
                       Accelerated Estimators
                  (Hot-patched via Intel sklearnex)
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
           [ Ridge Reg ]   [ Rand Forest ]   [ MLP Net ]
                 │               │               │
                 └───────────────┬───────────────┘
                                 ▼
                    Multi-Model Performance Metrics
                       (RMSE, MAE, R2 on Test)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      Diagnostics Visualizations       Post-Hoc SHAP Explanations
     (Residuals & Learning Curve)    (Proxy Model; max_depth=15 surrogate)
```

### 1. Data Splitting & Leakage Firewall
To establish an unbiased estimation of generalization performance, the 5,773 records are partitioned into a **70% Training Set (4,041 postings)** and a **30% Testing Set (1,732 postings)** with a fixed random seed (`random_state=42`). 
The `TfidfVectorizer` is fit *exclusively* on the training set, and the testing partition is only transformed using the training vocabulary. This prevents vocabulary leakage.

### 2. Custom Blacklist Stopwords Filtering (Noise Mitigation)
To prevent feature dilution and restore model accuracy, the pipeline shifts from an aggressive whitelist (which dropped crucial seniority indicators like `senior`, `lead`, `manager`) to an explicit custom blacklist stopword configuration. The custom blacklist is unioned with Scikit-Learn's `ENGLISH_STOP_WORDS` to filter out:
*   **German Leakage:** `und`, `mit`, `wir`, `der`, `du`, `die`, `den`, `auf`, `für`, `im`, `von`
*   **HR/Corporate Boilerplate:** `work`, `team`, `role`, `experience`, `build`, `world`, `solution`
*   **Domain & HTML Residues:** `cut edge`, `grocery`, `food`, `br`, `li`, `ul`, `div`

### 3. Model Architecture & Theoretical Justification
1.  **Ridge Regression:** Linear regression with L2 shrinkage penalty. Prevents overfitting on high-dimensional text vectors.
2.  **Random Forest Regressor:** Bagging ensemble decision tree regressor. Suited for modeling non-linear interactions among technical keyword combinations.
3.  **Multi-Layer Perceptron (MLP) Regressor:** Feedforward connectionist neural net topology. Fitted with Adam optimizer and early stopping logic.

### 4. Cross-Validation & Tuning Grids
Hyperparameters were tuned systematically using K-Fold cross-validation search wrappers:
*   **Ridge:** Grid search over $\alpha \in [0.01, 0.1, 1.0, 10.0, 100.0]$ (5-Fold CV). Best: `alpha = 1.0`.
*   **Random Forest:** Randomized search over tree count, split thresholds, and depth (3-Fold CV). Best: `n_estimators=100`, `min_samples_split=2`, `max_depth=None`.
*   **MLP:** Randomized search over hidden layer dimensions, activations, and L2 regularization weight (3-Fold CV). Best: `hidden_layer_sizes=(50, 25)`, `activation='relu'`, `alpha=0.0001`.

### 5. Quantitative Model Comparison
Predictive results evaluated on the unseen testing set ($X_{\text{test}}$), mapping log-transformed targets back to USD ($y_{\text{pred}} = \exp(y_{\text{pred\_log}}) - 1$):

| Model Architecture | Test RMSE ($) | Test MAE ($) | Test R-Squared ($R^2$) |
| :--- | :---: | :---: | :---: |
| **Ridge Regression** | \$37,363.68 | \$21,155.17 | 0.1176 (11.76%) |
| **Random Forest (Champion)** | **\$34,859.93** | **\$18,424.66** | **0.2319 (23.19%)** |
| **MLP Regressor (Neural Net)** | \$40,557.28 | \$23,597.36 | -0.0397 (-3.97%) |

*   **Random Forest** emerged as the champion model, explaining **30.21%** of variance by retaining critical seniority and context cues.
*   **MLP Regressor** underperformed, yielding a negative $R^2$. This demonstrates the limits of feedforward neural networks when operating directly on high-dimensional sparse TF-IDF vectors without dense word embeddings.

### 6. Diagnostics & Interpretability
*   **Residual Analysis:** scatter plots exhibit a diagonal trend, showing the models struggle to predict extreme salaries (roles paying above \$200,000) and tend to underpredict them, compressing residuals.
*   **Learning Curves:** The Random Forest training RMSE remains very low while cross-validation RMSE plateaus, indicating that text keywords explain a portion of salary variance, but geographic location and company size remain critical unmodeled features.
*   **SHAP Interpretability (Proxy Model Solution):** Computing SHAP values on deep unconstrained trees (`max_depth=None`) with sparse TF-IDF matrices triggered floating-point accumulation overflows in SHAP's C++ backend. To resolve this, we trained a **Surrogate Proxy Random Forest Regressor** with depth constrained to `max_depth=15`. This proxy replicates global decision boundaries while preventing additivity assertion overflow.
*   **SHAP Findings:** Concepts such as `engineering architecture` and `cloud` command the highest positive salary premiums, whereas generic operational keywords such as `delivery` reduce predicted salary values.

---

## 🚀 How to Run (Quick Start)

Replicate the environment and run both phases using PowerShell/Bash:

```bash
# Clone repository
git clone <repo-url>
cd <repo-folder>

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Phase 1 descriptive analytics
jupyter nbconvert --to notebook --execute descriptive_text_analytics.ipynb --ExecutePreprocessor.timeout=600

# Run Phase 3 machine learning modeling and diagnostics
jupyter nbconvert --to notebook --execute "(a3)(a2)machine_learning_modelling.ipynb" --ExecutePreprocessor.timeout=1200
```

---

## 📬 Contact
*   **Author:** Andy Deris
*   **Email:** andyderis36@gmail.com
*   **Affiliation:** School of Computing, Universiti Utara Malaysia (UUM)
