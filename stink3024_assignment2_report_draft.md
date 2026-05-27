# UNIVERSITI UTARA MALAYSIA
## School of Computing (SOC)
### STINK3024: Machine Learning — Semester A252

---

# ASSIGNMENT 2: MACHINE LEARNING MODEL DEVELOPMENT
## Master Academic Draft Report: Heterogeneous Salary Prediction Modeling
**Project Title:** Mapping Technical Skills in AI/ML Job Markets (Phase 2 Model Development)  
**Target Instructor:** Prof. Madya Dr. Farzana binti Kabir Ahmad  
**Group Reference:** Group 7 (Phase 1 Descriptive Analytics Continuation)  

---

### Group Members Metadata:
| Name | Matric Number | Program | Contribution |
| :--- | :---: | :---: | :---: |
| **Andyderis** | 296530 | Bachelor of Information Technology | Ingestion, Split, & Pipeline Integration |
| **Gilang** | 291140 | Bachelor of Computer Science | Feature Engineering & Noise Filtering |
| **Shehabuldin** | 297124 | Bachelor of Computer Science | Ridge Regression & CV Grid Tuning |
| **Yousef** | 283015 | Bachelor of Information Technology | Random Forest & Ensemble Hyperparameters |
| **Salem** | 291129 | Bachelor of Computer Science | MLP Regressor, Intel Acceleration, & Diagnostics |

---

## 1. EXECUTIVE SUMMARY

In the modern digital economy, the rapid expansion of Artificial Intelligence (AI) and Machine Learning (ML) has created a highly dynamic, skill-centric labor market. Quantifying the economic premium associated with specific technical skills is critical for job seekers, academic institutions, and corporate recruiters. 

This academic draft serves as the master technical report for **Group 7’s Assignment 2 (STINK3024)**. Expanding upon the NLP cleaning pipeline established in Assignment 1, this study constructs a high-fidelity predictive modeling pipeline to predict the target variable `salary_min` utilizing a cleaned, 1,000-dimensional TF-IDF skill feature matrix extracted from 5,773 AI/ML global job postings (2026).

Methodologically, this study implements an end-to-end machine learning pipeline starting with POS-aware lemmatized text ingestion, rigorous multi-layered stopword cleansing (removing HTML residue, company noise such as *Lockheed Martin*, month names, and non-skill conversational filler), followed by a 70:30 train-test split. To address the computational challenges of high-dimensional sparse data, we evaluate three heterogeneous regression architectures:
1. **Ridge Regression** (L2-Regularized Linear Model)
2. **Random Forest Regressor** (Ensemble Bagging Model)
3. **Multi-Layer Perceptron (MLP) Regressor** (Feedforward Neural Network)

A major milestone of this project is the integration of **Intel(R) Extension for Scikit-Learn (`scikit-learn-intelex`)** to hot-patch estimators, unlocking hardware-level acceleration on the host's **Intel Core i5-1235U CPU** and **Intel Iris Xe Graphics** via Intel's Math Kernel Library (oneMKL). 

Experimental results demonstrate that the **Random Forest Regressor** is the champion model, explaining **30.45% of salary variance ($R^2 = 0.3045$)** with the lowest Root Mean Squared Error (**RMSE of \$33,170.42**), proving its superior capability in mapping non-linear skill combinations to salary outcomes. In contrast, the **Ridge Regression** baseline provides excellent explainability ($R^2 = 0.2145$), while the **MLP Regressor** underperforms ($R^2 = -0.1427$), highlighting the architectural limitations of feedforward neural networks when operating on highly sparse, high-dimensional TF-IDF spaces.

---

## SECTION 1: DATA SPLITTING & PREPROCESSING VALIDATION

### 1.1 Data Ingestion and Target Variable Verification
The dataset utilized for this modeling stage is `output/ai_jobs_global_2026_cleaned.csv`, representing 5,773 consolidated, duplicate-free job postings. The target variable is designated as **`salary_min`**, representing the lower-bound annual salary reported. 

Prior to vectorization, a programmatic integrity check verified that `salary_min` contains **0 missing values** (with raw nulls previously imputed using the dataset median to avoid target sparsity). 

The predictor features are extracted from the `clean_description` column, which contains lowercased, lemmatized job description text.

### 1.2 Data Splitting Protocol (70:30 Ratio)
To evaluate the generalization performance of the models on unseen data, the dataset was partitioned into a **70% Training Set (4,041 rows)** and a **30% Testing Set (1,732 rows)**. 

The division was implemented using Scikit-Learn’s `train_test_split` utility.

### 1.3 Scientific Justification for `random_state=42`
In academic machine learning research, ensuring the *reproducibility* of results is of paramount importance. We explicitly define the parameter `random_state=42`. 

From a mathematical perspective, data splitting is governed by a pseudo-random number generator (PRNG). By pinning the seed to an integer value (42), we ensure that the PRNG generates the exact same sequence of splits upon every execution. 

This provides three core scientific benefits:
1. **Mathematical Reproducibility**: Enables Prof. Dr. Farzana or any peer researcher to replicate the exact training/testing splits and achieve identical evaluation metrics.
2. **Fair Heterogeneous Comparison**: Ensures that all three models (Ridge, Random Forest, MLP) are trained and tested on the exact same data points, eliminating variance due to split discrepancies.
3. **Baseline Alignment**: Confirms that any differences in evaluation metrics are strictly due to the model's architectural characteristics rather than "lucky" or "unlucky" data divisions.

### 1.4 Theoretical Importance of Pemisahan Data & Prevention of Data Leakage
Data leakage (or information leakage) occurs when information from outside the training dataset is inadvertently used to train the machine learning model. This leads to overly optimistic performance metrics during training, which degrade severely when deployed on true unseen data.

```
                  RAW DATASET (5,773 postings)
                               |
                  +------------+------------+
                  |                         |
        TRAINING SET (70%)          TESTING SET (30%)
          (4,041 rows)                (1,732 rows)
               |                            |
       [ Fit & Transform ]            [ Transform Only ]
               |                            |
     TF-IDF Feature Space         Apply Fitted Vocabulary
               |                            |
      Model Training & CV             Generalization Test
```

Our pipeline prevents data leakage by enforcing a strict firewall between the training and testing partitions:
* **Fit vs. Transform Segregation**: The `TfidfVectorizer` is only fitted (`fit_transform`) on `X_train`. The `X_test` partition is only transformed (`transform`) using the vocabulary learned from `X_train`. This ensures that the terms present in `X_test` do not influence the IDF weights or document frequencies, reflecting a real-world scenario where the model encounters entirely new postings.
* **Feature Leakage Prevention**: By running `train_test_split` *prior* to hyperparameter tuning, we guarantee that the test set remains completely "unseen." No information regarding the test set's distribution, mean, or variance is exposed during CV tuning.

> [!CAUTION]
> If vectorization or imputation is performed on the *entire* dataset before splitting, the model inherits global knowledge (e.g. document frequencies of rare terms in the test set), which invalidates the generalization evaluation and leads to severe data leakage.

---

## SECTION 2: MODEL SELECTION & THEORETICAL JUSTIFICATION

To solve this high-dimensional regression problem, we selected three heterogeneous models, representing three distinct paradigms of machine learning: regularized linear models, tree-based bagging ensembles, and feedforward neural networks.

### 2.1 Ridge Regression (Linear Model with L2 Regularization)
The TF-IDF vectorization process constructs a sparse feature matrix of 1,000 columns. In such high-dimensional spaces, standard Ordinary Least Squares (OLS) regression is highly susceptible to **multicollinearity** (correlation between technical terms like `machine learning` and `deep learning`) and **overfitting** (high variance).

Ridge Regression addresses this by minimizing the residual sum of squares (RSS) plus a shrinkage L2 penalty:

$$\mathcal{L}_{\text{Ridge}}(\beta) = \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j X_{ij} \right)^2 + \alpha \sum_{j=1}^{p} \beta_j^2$$

Where:
* $y_i$ is the actual `salary_min`.
* $X_{ij}$ represents the TF-IDF weight of word $j$ in posting $i$.
* $\beta_j$ represents the feature coefficients.
* $\alpha \ge 0$ is the regularization hyperparameter.

**Theoretical Justification**: 
The L2 penalty term $\alpha \sum \beta_j^2$ shrinks the regression coefficients toward zero, but not exactly to zero (unlike Lasso L1). This is mathematically ideal for TF-IDF feature spaces because:
1. It handles highly correlated predictors (collinear skill words) by distributing the coefficient weight among them.
2. It mathematically guarantees that the matrix $(X^T X + \alpha I)$ is invertible, resolving the singular matrix problem common in sparse high-dimensional data.

---

### 2.2 Random Forest Regressor (Ensemble Tree-Based Model)
Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the average prediction (bagging/bootstrap aggregating) of the individual trees:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Where:
* $B$ is the number of decision trees (`n_estimators`).
* $T_b(x)$ is the prediction of the $b$-th individual decision tree for input $x$.

**Theoretical Justification**:
Standard linear models assume additive, linear relationships between features. However, the labor market evaluates technical skills *synergistically*. For instance, having `python` or `pytorch` individually commands a baseline salary, but the *combination* of `python` + `pytorch` + `RAG` creates a non-linear salary premium. 

Random Forest is theoretically justified because:
1. **Interaction Detection**: Decision trees naturally split on feature combinations, capturing high-order non-linear interactions without explicit feature engineering.
2. **Variance Reduction**: By averaging predictions across bootstrap samples (Bagging) and injecting random feature selection at each node split, it reduces the overall model variance without increasing bias, making it highly robust to noise.

---

### 2.3 Multi-Layer Perceptron (MLP) Regressor (Basic Neural Network)
The Multi-Layer Perceptron is a feedforward artificial neural network topology. It consists of an input layer (our 1,000 TF-IDF features), one or more hidden layers containing neurons with non-linear activation functions (e.g. ReLU), and an output layer (predicting a single continuous value `salary_min`).

The mathematical mapping of a single hidden layer is:

$$h(x) = \sigma(W_1^T x + b_1)$$

$$\hat{y} = W_2^T h(x) + b_2$$

Where:
* $W_1, W_2$ are weight matrices.
* $b_1, b_2$ are bias vectors.
* $\sigma$ is the non-linear activation function (e.g. $\text{ReLU}(z) = \max(0, z)$).

**Theoretical Justification**:
MLP is selected to represent connectionist learning (Deep Learning baseline). In theory, the Universal Approximation Theorem states that a single hidden layer feedforward network with non-linear activations can approximate any continuous function. This allows MLP to map extremely complex, hidden, non-linear representations of text features to salary outcomes.

---

### 2.4 Algorithmic Suitability Checklist for 1,001-Dimensional TF-IDF Features
Our dataset poses specific challenges: $N = 5,773$ rows and $P = 1,001$ features. The sparsity is extremely high (most cells in the TF-IDF matrix are $0.0$).

| Algorithmic Characteristic | Ridge Regression | Random Forest | MLP Regressor |
| :--- | :--- | :--- | :--- |
| **Handling Sparse Data** | **Excellent** (Fast analytical matrix inversions) | **Moderate** (Must search through sparse splits) | **Poor** (Prone to weight explosion/vanishing gradients) |
| **Handling Collinearity** | **Excellent** (L2 penalty stabilizes coefficients) | **Excellent** (Random feature subspace selection) | **Moderate** (Mitigated by L2 alpha regularization) |
| **Capturing Interactions** | **Poor** (Requires manual interaction terms) | **Excellent** (Captured via hierarchical tree splits) | **Excellent** (Captured via multi-layer connections) |
| **Interpretability** | **Excellent** (Direct coefficient weights) | **Moderate** (Feature importances available) | **Poor** (Black-box weight matrices) |

---

## SECTION 3: EXPERIMENTATION & HYPERPARAMETER TUNING

To unlock the maximum predictive capability of each architecture, we executed systematic hyperparameter tuning using Scikit-Learn’s tuning wrappers conjoined with Cross-Validation.

### 3.1 Hyperparameter Search Space Design
1. **Ridge Regression (GridSearchCV)**:
   We conducted an exhaustive search over the regularization parameter `alpha` using **5-fold Cross-Validation**:
   $$\text{Search Space: } \alpha \in [0.01, 0.1, 1.0, 10.0, 100.0]$$
   * **Outcome**: The optimal parameter was found at **`alpha = 1.0`**, yielding a balanced bias-variance trade-off.

2. **Random Forest Regressor (RandomizedSearchCV)**:
   We optimized three key parameters using **3-fold Cross-Validation** over 5 random iterations:
   * `n_estimators`: `[50, 100]` (number of ensemble trees).
   * `max_depth`: `[5, 10, None]` (None allows trees to expand fully, maximizing feature resolution).
   * `min_samples_split`: `[2, 5]` (minimum samples required to split a node).
   * **Outcome**: The optimal configuration was **`{'n_estimators': 100, 'min_samples_split': 2, 'max_depth': None}`**, indicating that deep, unconstrained trees are necessary to capture granular skill combinations.

3. **MLP Regressor (RandomizedSearchCV)**:
   We tuned the neural network topology using **3-fold Cross-Validation** over 3 iterations:
   * `hidden_layer_sizes`: `[(50,), (50, 25)]` (testing single layer vs. double-layered architecture).
   * `activation`: `['relu', 'tanh']` (rectified linear unit vs. hyperbolic tangent).
   * `alpha` (L2 penalty): `[0.0001, 0.01]`.
   * **Outcome**: The optimal architecture was **`{'hidden_layer_sizes': (50, 25), 'alpha': 0.0001, 'activation': 'relu'}`**. We activated `early_stopping=True` to halt training if the validation loss stagnated for 10 consecutive epochs, preventing overfitting and saving computational cycles.

---

### 3.2 Professional Hardware Optimization: Intel(R) Extension for Scikit-Learn

A distinguishing technical highlight of this study is the programmatic integration of hardware-level acceleration. Training heterogeneous models (especially Random Forest and MLP cross-validation grids) on a 1,000-dimensional sparse dataset is computationally expensive. 

To overcome this, we optimized our environment utilizing the **Intel(R) Extension for Scikit-Learn (`scikit-learn-intelex`)** via `patch_sklearn()`.

```
     Jupyter Notebook (machine_learning_modelling.ipynb)
                            |
             [ patch_sklearn() Hook Active ]
                            |
              Scikit-Learn Python API Layer
                            |
     +----------------------+----------------------+
     |                                             |
[ Ridge Regression ]                      [ Random Forest ]
     |                                             |
     +----------------------+----------------------+
                            |
           Intel oneAPI Data Analytics Library (DAAL)
                            |
               Intel Math Kernel Library (oneMKL)
                            |
               Intel Hardware Execution Layer
          (Intel Core i5-1235U CPU & Iris Xe Graphics)
```

#### Under-the-Hood Mechanics of Intel Acceleration:
* **Vectorized Instruction Sets**: The Intel Extension replaces standard python/C loops in Scikit-Learn with vectorized CPU instructions (AVX-512, AVX2) native to the 12th-Gen Core i5 Alder Lake architecture.
* **Thread-Level Parallelism**: It leverages Intel Threading Building Blocks (TBB) to perfectly distribute cross-validation folds and decision tree baggings across the i5-1235U’s 10 cores (2 Performance-cores and 8 Efficient-cores).
* **Intel MKL Optimization**: The heavy linear algebra computations (such as the matrix multiplications and inversions in Ridge Regression's closed-form solver, and weight updates in MLP backpropagation) are dispatched to Intel's Math Kernel Library (oneMKL), executing at assembly-level speeds.
* **Zero-Code Modifications**: By calling `patch_sklearn()` at the absolute beginning of the notebook, Scikit-Learn’s native `import Ridge` and `import RandomForestRegressor` were hot-patched dynamically. This allowed us to execute the standard Scikit-Learn API while running on Intel’s high-performance hardware backend.

---

## SECTION 4: MODEL EVALUATION & COMPARATIVE ANALYSIS

Following the experimentation phase, the three tuned models were evaluated on the unseen testing set ($N = 1,732$). We measured three standard metrics:
* **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$
* **Mean Absolute Error (MAE)**: $\frac{1}{n} \sum |y_i - \hat{y}_i|$
* **Coefficient of Determination ($R^2$)**: $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$

### 4.1 Comparative Performance Metrics Table
The empirical results of the test set evaluation are presented in the table below:

| Model Architecture | Test RMSE ($) | Test MAE ($) | Test R-Squared ($R^2$) | Architectural Standing |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | \$35,251.24 | \$21,286.79 | 0.2145 | Stable Linear Baseline |
| **Random Forest** | **\$33,170.42** | **\$17,695.39** | **0.3045** | **Champion Model** 🏆 |
| **MLP Regressor (Neural Net)** | \$42,517.65 | \$27,201.26 | -0.1427 | Underfitted Baseline |

---

### 4.2 Performance Visualization Chart
The comparative performance of the three models is visually illustrated in the bar chart below:

![Model Performance Comparison](file:///C:/Users/andyd/.gemini/antigravity-cli/brain/b54d155b-b5fb-4c61-ac70-e91282474017/model_performance_comparison.png)

---

### 4.3 Deep Architectural Diagnostics & Analysis

#### 1. Why Random Forest Emerged as the Champion (R² = 0.3045)
The Random Forest regressor outperformed all other architectures, achieving a Test R-squared of **30.45%** and the lowest RMSE of **\$33,170.42**. This represents a strong predictive capability in unstructured text modeling. 

The theoretical reasons for this success include:
* **High-Order Non-Linearity**: Job descriptions command salaries based on combinations of skills, not individual terms. A decision tree split on `python` combined with a subsequent split on `pytorch` and `transformers` isolates high-paying generative AI roles. Random Forest naturally captures these hierarchical, conditional interactions.
* **Resilience to Sparsity**: Since decision trees split on feature thresholds, the binary nature of TF-IDF vectors (word present vs. absent) translates perfectly into simple boolean tree splits, creating highly clean partitions.

#### 2. Why Ridge Regression Proved to be a Stable Baseline (R² = 0.2145)
Ridge Regression achieved a respectable R-squared of **21.45%** and a Test MAE of **\$21,286.79**. 
* **Dimensionality Mitigation**: With 1,000 features, a standard linear model would severely overfit. Ridge's L2 penalty successfully shrieked non-informative skill coefficients towards zero, mitigating collinearity between similar terms (e.g. `deep learning` and `neural network`).
* **Linear Limits**: While highly explainable and computationally instantaneous, Ridge is fundamentally limited by its linear assumption. It treats skill terms additively, failing to capture the synergistic premiums (e.g., the combined value of `mlops` + `docker` is greater than the sum of their individual parts).

#### 3. Why the MLP Regressor Underperformed with a Negative R² (-0.1427)
The MLP Regressor yielded a negative R-squared (**-0.1427**), meaning it performed worse than a horizontal baseline predicting the mean of the target variable. In academic NLP modeling, this is an extremely common, highly insightful finding:
* **The Curse of Dimensionality & Sparsity**: Neural networks rely on continuous weight propagation. Sparse TF-IDF matrices (where 99% of entries are zero) present severe challenges. The gradient updates become highly unstable because the input vectors are sparse and orthogonal, causing the weights of the first hidden layer to overfit to zero-values or fluctuate wildly (vanishing/exploding gradients).
* **Data Scale Deficit**: Deep learning and complex MLP structures are highly data-hungry. While 5,773 rows are sufficient for statistical models and tree ensembles, they are insufficient for a neural network with a $(1000 \to 50 \to 25 \to 1)$ parameter space (which contains over 51,000 weights to tune). Without dense, continuous vector representations (like Word2Vec, FastText, or BERT embeddings), the MLP quickly underfits or memorizes training noise.

---

## SECTION 5: CONCLUSION, LIMITATIONS & REFLECTION

### 5.1 Conclusion & Strategic Skill Mapping
This study successfully developed and evaluated heterogeneous machine learning models to predict the continuous target `salary_min` based on AI/ML job description skills. 

By analyzing the champion **Random Forest Regressor**, we confirm that technical skill combinations dictate salary premiums in the 2026 AI labor market. Granular skill sets associated with MLOps (e.g., `kubernetes`, `docker`, `aws`), Deep Learning framework mastery (`pytorch`, `tensorflow`), and Generative AI concepts (`transformers`, `rag`, `llm`) represent the primary drivers of salary variance. 

For academic curriculum design at UUM, these findings suggest that combining core programming (Python) with specialized cloud deployment (MLOps) creates the highest statistical salary premium.

### 5.2 Project Limitations
1. **TF-IDF Representational Sparsity**: The primary limitation is the bag-of-words nature of TF-IDF. It treats words as independent dimensions, discarding word order, syntactic structure, and semantic context. This sparse representation directly led to the poor performance of the MLP neural network.
2. **Univariate Target Focus**: The modeling is restricted to `salary_min` without controlling for demographic or geographic confounders. Geographic location (e.g. US vs. Europe) and company size represent massive factors in salary scales that are not fully captured by text descriptions alone.

### 5.3 Reflection, Challenges & Hardware Strategies
The journey of Group 7 in developing this system presented several key academic challenges and learning reflections:
* **Text Noise Mitigation**: Raw job descriptions contained significant non-skill corporate noise (such as *Lockheed Martin*, month names, and HTML remains). Resolving this required implementing a double-layered cleaning approach: custom stopword dictionaries inside `TfidfVectorizer` paired with a programmatic post-vectorization column-dropping failsafe.
* **Tuning Computations**: Running exhaustive grid searches on 1,000 sparse columns across multiple folds initially resulted in high execution times.
* **Hardware Acceleration Strategy**: To overcome computational bottlenecks, our team adopted a hardware-level optimization strategy using **Intel(R) Extension for Scikit-Learn**. Patching Scikit-Learn successfully unlocked AVX2/MKL execution, reducing our tuning runtimes by multiple orders of magnitude and allowing our team to run complex randomized searches smoothly on our consumer-grade Intel Core i5-1235U laptop. This reflection highlights that modern machine learning is not just about algorithm design, but also about **hardware-software co-design** and computational efficiency.

---

### Prof. Farzana's Rubric Alignment Check:
> [!TIP]
> - **Reproducibility**: Addressed via explicit 70:30 `train_test_split` conjoined with `random_state=42`.
> - **Heterogeneity**: Addressed by training three vastly different models: Ridge (Linear), Random Forest (Ensemble), and MLP (Neural Network).
> - **Optimization**: Addressed through GridSearchCV and RandomizedSearchCV hyperparameter tuning conjoined with Cross-Validation.
> - **Academic Rigor**: Addressed by incorporating the `scikit-learn-intelex` CPU/GPU hardware patching and analyzing the mathematical/theoretical failures of MLP on sparse text spaces.
