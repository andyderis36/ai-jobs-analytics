# Code Documentation: Advanced Modeling, Diagnostics, and SHAP Interpretability
**Notebook File:** [(a3)(a2)machine_learning_modelling.ipynb](file:///C:/Users/andyd/Documents/UUM/UUM%20OL/A252/ML/Assignment/(a3)(a2)machine_learning_modelling.ipynb)  
**Phase:** Assignment 3 (Final Modeling & Interpretability)  
**Language:** Python 3  

---

## 1. Custom Blacklist Stopword Filtering

This notebook introduces a comprehensive blacklist stopword configuration to remove German language leakage, generic HR fillers, and domain-specific noise while retaining key markers of seniority and business context.

```python
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Split data first
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_description'], df['salary_min'], test_size=0.30, random_state=42
)

# Target log transformation
y_train_log = np.log1p(y_train)

# 2. Comprehensive custom blacklist definition
custom_blacklist = [
    # German Leakage
    'und', 'mit', 'wir', 'der', 'du', 'die', 'den', 'auf', 'für', 'im', 'von',
    # HR/Corporate Fillers
    'work', 'team', 'role', 'experience', 'build', 'world', 'solution', 'job', 'company', 'customer', 'join', 'opportunity', 'description', 'position', 'mission', 'people', 'year', 'time', 'look', 'support', 'client', 'real', 'requirement', 'grow', 'global', 'love', 'apply', 'life', 'problem', 'delivery', 'qualification', 'organization', 'skill', 'quality', 'environment', 'professional', 'partner', 'day', 'care', 'candidate', 'health', 'believe', 'ability', 'serve', 'share', 'highly', 'end', 'enable', 'operation', 'employee', 'improve', 'meet', 'shape', 'require', 'offer', 'user', 'process', 'responsibility', 'decision', 'information', 'group', 'connect', 'management', 'bring', 'responsible', 'access', 'enjoy', 'fast', 'hour', 'office', 'ensure', 'million', 'strong', 'culture', 'collaborate', 'standard', 'member', 'value', 'salary', 'range', 'include', 'deliver', 'seek',
    # Domain Noise & HTML Leftovers
    'cut edge', 'cut', 'edge', 'grocery', 'food', 'job description', 'grade', 'br', 'li', 'ul', 'div', 'span', 'href', 'html', 'amp'
]

# 3. Stop-words Unioning
from sklearn.feature_extraction import text
final_stop_words = list(text.ENGLISH_STOP_WORDS.union(custom_blacklist))
```

---

## 2. TF-IDF Feature Engineering

Converts text inputs into numerical TF-IDF feature matrices using the combined custom stopwords.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Re-initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=10,
    max_df=0.80,
    max_features=1000,
    stop_words=final_stop_words
)

# fit_transform on X_train, transform on X_test
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names_out()
```

---

## 3. Multi-Model Residual Diagnostics

Calculates prediction residuals to check for homoscedasticity and target bias.

```python
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
models_dict = {
    'Ridge Regression': best_ridge, 
    'Random Forest': best_rf, 
    'MLP Regressor': best_mlp
}

for idx, (name, model_obj) in enumerate(models_dict.items()):
    pred_log = model_obj.predict(X_test_tfidf)
    pred = np.expm1(pred_log)
    residuals = y_test - pred
    
    sns.scatterplot(x=pred, y=residuals, ax=axes[idx], alpha=0.5, color='teal')
    axes[idx].axhline(y=0, color='r', linestyle='--')
    axes[idx].set_title(f'Residual Plot: {name}', fontweight='bold')
    axes[idx].set_xlabel('Predicted Salary ($)')
    axes[idx].set_ylabel('Residuals (Actual - Predicted)')

plt.suptitle('Model Residual Analysis on Test Data', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output_modelling/residual_plots.png', dpi=300)
```

*   **Interpretation:** Checks that residuals are randomly distributed around $y=0$. The diagonal boundary pattern indicates the models struggle to predict extreme high-salary outliers ($> \$200,000$).

---

## 4. Learning Curves (Generalization Diagnostics)

Evaluates the fitting behavior of the champion Random Forest model across training sample sizes.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    best_rf, X_train_tfidf, y_train_log, 
    cv=3, 
    scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 5),
    random_state=42
)

# Convert MSE to RMSE
train_rmse = np.sqrt(-train_scores)
val_rmse = np.sqrt(-val_scores)

train_mean = np.mean(train_rmse, axis=1)
train_std = np.std(train_rmse, axis=1)
val_mean = np.mean(val_rmse, axis=1)
val_std = np.std(val_rmse, axis=1)

# Plotting curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training RMSE')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 's-', color='orange', label='Cross-Validation RMSE')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
plt.title('Learning Curve: Random Forest Regressor', fontsize=14, fontweight='bold')
plt.xlabel('Number of Training Samples')
plt.ylabel('RMSE (Log Scale)')
plt.legend(loc='best')
plt.grid(True, linestyle='--')
plt.savefig('output_modelling/learning_curve_rf.png', dpi=300)
```

---

## 5. Gini Feature Importances

Extracts the relative contribution of each technical term to prediction error reduction.

```python
importances = best_rf.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(data=rf_importance_df, x='Importance', y='Feature', palette='magma')
plt.title('Top 20 Technical Skills (Random Forest Importances)', fontweight='bold')
plt.xlabel('Gini Importance')
plt.ylabel('Skill Feature Term')
plt.savefig('output_modelling/rf_feature_importances.png', dpi=300)
```

---

## 6. SHAP Interpretability & Surrogate Proxy Model

SHAP TreeExplainer computes additive local feature contributions. However, computing SHAP values on deep decision trees (`max_depth=None`) with sparse TF-IDF vectors can cause floating-point accumulation overflows in SHAP's C++ backend, triggering additivity assertion errors.

To address this, the notebook implements a **Surrogate Proxy Model** with constrained depth.

```python
import sys
import shap
from sklearnex import unpatch_sklearn

# 1. Unpatch sklearnex to restore standard Scikit-Learn classes in memory
unpatch_sklearn()

# 2. Reload ensemble modules from sys.modules
to_del = [m for m in list(sys.modules.keys()) if m.startswith('sklearn.ensemble')]
for m in to_del:
    del sys.modules[m]

from sklearn.ensemble import RandomForestRegressor

# 3. Fit a Surrogate Proxy Random Forest with max_depth constrained
proxy_params = best_rf_params.copy()
proxy_params['max_depth'] = 15 # Constrained depth prevents floating-point overflow

proxy_rf = RandomForestRegressor(**proxy_params, random_state=42)
proxy_rf.fit(X_train_tfidf, y_train_log)

# 4. Calculate SHAP values on test sample
np.random.seed(42)
sample_indices = np.random.choice(X_test_tfidf.shape[0], size=min(X_test_tfidf.shape[0], 30), replace=False)
X_test_sample = X_test_tfidf[sample_indices]

explainer = shap.TreeExplainer(proxy_rf)
shap_values = explainer.shap_values(X_test_sample.toarray())

# 5. Visualize and save summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, 
    X_test_sample.toarray(), 
    feature_names=feature_names, 
    show=False
)
plt.title("SHAP Summary Plot: Feature Impact on Salary", fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('output_modelling/shap_summary_plot.png', dpi=300)
```
