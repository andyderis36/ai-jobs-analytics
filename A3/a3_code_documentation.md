# Code Documentation: Advanced Modeling, Diagnostics, and SHAP Interpretability
**Notebook File:** [(a3)machine_learning_modelling.ipynb](file:///C:/Users/andyd/Documents/UUM/UUM%20OL/A252/ML/Assignment/(a3)machine_learning_modelling.ipynb)  
**Phase:** Assignment 3 (Final Modeling & Interpretability)  
**Language:** Python 3  

---

## 1. Domain-Specific Technical Filter

This notebook introduces a regex-based whitelist filter to clean the text corpus prior to vectorization, isolating technical skills and terminology.

```python
import re
from sklearn.model_selection import train_test_split

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_description'], df['salary_min'], test_size=0.30, random_state=42
)

# 140+ core technical keywords whitelist
strict_tech_keywords = {
    'ai', 'ml', 'data', 'engineer', 'engineering', 'software', 'technology', 'model', 
    'analytics', 'python', 'architecture', 'cloud', 'developer', 'development', 
    'system', 'solution', 'science', 'infrastructure', 'design', 'platform', 
    'algorithm', 'intelligence', 'artificial', 'deep', 'learning', 'machine',
    'aws', 'azure', 'sql', 'api', 'vision', 'nlp', 'language', 'llm', 'llms',
    'generative', 'genai', 'prompt', 'rag', 'agent', 'agentic', 'robotics',
    'automation', 'pipeline', 'deployment', 'mlops', 'framework', 'devops',
    'backend', 'frontend', 'fullstack', 'database', 'warehouse', 'big',
    'spark', 'hadoop', 'kubernetes', 'docker', 'container', 'linux', 'bash',
    'git', 'testing', 'security', 'network', 'hardware', 'gpu', 'tpu',
    'tensorflow', 'pytorch', 'keras', 'scikit', 'pandas', 'numpy', 'scipy',
    'tableau', 'powerbi', 'dashboard', 'visualization', 'reporting', 'bi',
    'statistics', 'math', 'mathematics', 'optimization', 'predictive', 'analysis',
    'analyst', 'scientist', 'architect', 'research', 'researcher', 'lab',
    'java', 'c++', 'javascript', 'typescript', 'react', 'node', 'go', 'rust',
    'ruby', 'php', 'html', 'css', 'nosql', 'mongodb', 'cassandra', 'redis',
    'elasticsearch', 'snowflake', 'databricks', 'airflow', 'mlflow', 'kubeflow',
    'sagemaker', 'vertex', 'huggingface', 'langchain', 'openai', 'gpt', 'bert',
    'llama', 'claude', 'anthropic', 'meta', 'google', 'microsoft', 'amazon',
    'scale', 'scalable', 'production', 'training', 'inference', 'finetuning',
    'dataset', 'datasets', 'annotation', 'labeling', 'computer', 'cognitive',
    'neural', 'networks', 'transformer', 'transformers', 'diffusion', 'audio',
    'video', 'image', 'speech', 'text', 'processing', 'mining', 'scraping',
    'web', 'app', 'application', 'mobile', 'ios', 'android', 'ui', 'ux',
    'agile', 'scrum', 'jira', 'github', 'gitlab', 'terraform', 'gcp',
    'microservices', 'metrics', 'query', 'mysql', 'postgresql', 'oracle',
    'server', 'serverless', 'technical', 'technologies', 'code', 'coding',
    'programming', 'programmer', 'scripting', 'script', 'maths', 'statistical',
    'modeling', 'models', 'algorithms', 'quantitative', 'quant', 'cv', 'dl',
    'integration', 'continuous', 'delivery', 'cicd', 'unix', 'ubuntu',
    'debian', 'centos', 'redhat', 'windows', 'macos', 'jupyter', 'notebook',
    'colab', 'kaggle'
}

def domain_specific_filter(text):
    """Retain only tokens present in the technical whitelist"""
    text = str(text).lower()
    # Extract words/tokens
    tokens = re.findall(r'\b[a-z+-]+\b', text)
    filtered_tokens = [t for t in tokens if t in strict_tech_keywords]
    return " ".join(filtered_tokens)

# Apply filter
X_train_filtered = X_train.apply(domain_specific_filter)
X_test_filtered = X_test.apply(domain_specific_filter)
```

*   **Logic:** Non-technical words, punctuation, and digits are removed. This restricts the vectorizer's vocabulary to technical terms, reducing dimensionality and noise.

---

## 2. Multi-Model Residual Diagnostics

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

*   **Interpretation:** The diagnostic plot checks that residuals are randomly distributed around $y=0$. The diagonal boundary pattern indicates the models tend to underpredict extreme high-salary outliers ($> \$200,000$).

---

## 3. Learning Curves (Generalization Diagnostics)

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

*   **Logic:** Plots train and cross-validation RMSE as a function of the number of training samples, helping diagnose whether the model is underfitting or overfitting.

---

## 4. Gini Feature Importances

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

## 5. SHAP Interpretability & Surrogate Proxy Model

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

*   **Proxy Logic:**
    1.  `unpatch_sklearn()` removes the Intel runtime hooks.
    2.  `max_depth` is constrained to `15` to limit tree depth.
    3.  This surrogate model captures the global decision boundaries of the champion model while keeping paths shallow enough for SHAP's C++ backend to execute without floating-point overflow.
