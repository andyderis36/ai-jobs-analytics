# Code Documentation: Machine Learning Model Development & Tuning
**Notebook File:** [machine_learning_modelling.ipynb](file:///C:/Users/andyd/Documents/UUM/UUM%20OL/A252/ML/Assignment/machine_learning_modelling.ipynb)  
**Phase:** Assignment 2 (Model Development)  
**Language:** Python 3  

---

## 1. Hardware Acceleration & Verification

To accelerate model fitting across hyperparameter search grids, the notebook integrates the Intel(R) Extension for Scikit-Learn.

```python
# Enable Intel CPU math kernel optimizations
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Intel(R) Extension for Scikit-Learn successfully patched!")
except ImportError:
    print("Running standard Scikit-Learn.")

# Verification
from sklearnex import sklearn_is_patched
print("Is Ridge accelerated?", sklearn_is_patched('Ridge'))
print("Is RandomForest accelerated?", sklearn_is_patched('RandomForestRegressor'))
```

*   **Logic:** `patch_sklearn()` overrides Scikit-Learn algorithms (e.g., Ridge, Random Forest) with optimized, vectorized C++ implementations using Intel Math Kernel Library (oneMKL) instructions, accelerating computation on Intel architectures.

---

## 2. Ingestion & Pre-modeling Data Splits

The cleaned text output from Phase 1 is loaded. A strict data partition firewall is implemented.

```python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv(os.path.join('output', 'ai_jobs_global_2026_cleaned.csv'))
df['clean_description'] = df['clean_description'].fillna('')
df['salary_min'] = df['salary_min'].fillna(df['salary_min'].median())

# Extract features and target
X = df['clean_description']
y = df['salary_min']

# Partition: 70% Train, 30% Test with a fixed seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Apply log transformation to the target
y_train_log = np.log1p(y_train)
```

*   **`random_state=42`:** Pinning the PRNG seed ensures identical data splits across executions, making evaluations reproducible.
*   **Log Transformation (`np.log1p`):** Calculates $y_{\text{log}} = \ln(1 + y)$. This normalizes the skewed salary target, helping the algorithms converge.

---

## 3. TF-IDF Feature Engineering Firewall

The notebook translates text to numeric vectors while preventing information leakage.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize unigrams & bigrams
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85,
    max_features=1000
)

# Fit on training data ONLY; transform both sets
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names_out()
```

*   **Anti-Leakage Firewall:** `fit_transform` is called exclusively on `X_train`. The testing set `X_test` is only processed using `transform` to apply the learned vocabulary and document frequencies. This ensures terms from the test set do not influence the IDF calculations.

---

## 4. Hyperparameter Cross-Validation Grids

Tuning setups are defined for three heterogeneous architectures using K-Fold Cross-Validation.

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')
```

### 4.1 Ridge Regression Tuning (L2 Penalty)
Tuned using exhaustive 5-fold grid search:
```python
ridge_base = Ridge(random_state=42)
ridge_param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

ridge_grid = GridSearchCV(
    estimator=ridge_base,
    param_grid=ridge_param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
ridge_grid.fit(X_train_tfidf, y_train_log)
best_ridge = ridge_grid.best_estimator_
```

### 4.2 Random Forest Tuning (Bagging Ensemble)
Tuned using 3-fold randomized search over 5 iterations:
```python
rf_base = RandomForestRegressor(random_state=42)
rf_param_dist = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=rf_param_dist,
    n_iter=5,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42
)
rf_random.fit(X_train_tfidf, y_train_log)
best_rf = rf_random.best_estimator_
```

### 4.3 MLP Regressor Tuning (Neural Network)
Tuned using 3-fold randomized search over 3 iterations with early stopping active:
```python
mlp_base = MLPRegressor(
    solver='adam',
    max_iter=150,
    early_stopping=True,
    random_state=42
)
mlp_param_dist = {
    'hidden_layer_sizes': [(50,), (50, 25)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.01]
}

mlp_random = RandomizedSearchCV(
    estimator=mlp_base,
    param_distributions=mlp_param_dist,
    n_iter=3,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42
)
mlp_random.fit(X_train_tfidf, y_train_log)
best_mlp = mlp_random.best_estimator_
```

---

## 5. Evaluation & Comparison Metrics

Evaluates predictions on the unseen testing set ($X_{\text{test\_tfidf}}$) and maps predictions back to USD before metric calculation.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

models = {
    'Ridge Regression': best_ridge,
    'Random Forest': best_rf,
    'MLP Regressor': best_mlp
}

evaluation_results = []

for name, model in models.items():
    # Predict in log scale
    pred_log = model.predict(X_test_tfidf)
    # Map back to USD
    pred = np.expm1(pred_log)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    evaluation_results.append({
        'Model Architecture': name,
        'Test RMSE ($)': rmse,
        'Test MAE ($)': mae,
        'Test R-Squared (R2)': r2
    })

results_df = pd.DataFrame(evaluation_results)
results_df.to_csv('output_modelling/model_evaluation_metrics.csv', index=False)
```

*   **Log Inversion (`np.expm1`):** Inverts predictions back to the original USD scale ($y = e^{y_{\text{log}}} - 1$), ensuring calculated RMSE, MAE, and $R^2$ values are mathematically sound.
*   **Result Visualization:** Plots Test RMSE and $R^2$ as a side-by-side bar chart, saved to `output_modelling/model_performance_comparison.png`.
