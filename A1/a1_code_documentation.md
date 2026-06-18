# Code Documentation: Descriptive Text Analytics & Preprocessing
**Notebook File:** [descriptive_text_analytics.ipynb](file:///C:/Users/andyd/Documents/UUM/UUM%20OL/A252/ML/Assignment/descriptive_text_analytics.ipynb)  
**Phase:** Assignment 1 (Descriptive Text Analytics)  
**Language:** Python 3  

---

## 1. Module Imports & Environment Setup

This notebook begins by importing libraries required for data manipulation, natural language processing, numerical analysis, and plotting.

```python
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# NLTK components for NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Scikit-Learn vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

*   **`nltk` Downloads:** Programmatically downloads dictionaries and taggers (WordNet, perceptron tagger, and stopwords corpus) to guarantee execution regardless of the local environment state.
*   **Visual Themes:** Seaborn style is set to `whitegrid` with a default figure size of `(12, 6)` to establish a consistent theme across output charts.

---

## 2. Ingestion & Target Variable Alignment

The raw dataset `ai_jobs_global_2026.csv` is ingested. This phase cleans the target salary columns (`salary_min` and `salary_max`) and handles missing values.

```python
# Load the dataset
df = pd.read_csv('ai_jobs_global_2026.csv')

# Parse and clean salary fields
def clean_salary_value(val):
    if pd.isna(val):
        return np.nan
    # Remove currency symbol and comma separators
    cleaned = re.sub(r'[^\d\.]', '', str(val))
    return float(cleaned) if cleaned else np.nan

df['salary_min'] = df['salary_min'].apply(clean_salary_value)
df['salary_max'] = df['salary_max'].apply(clean_salary_value)

# Create unified average salary column
df['salary'] = df[['salary_min', 'salary_max']].mean(axis=1)

# Impute missing salary values with dataset median
median_salary = df['salary'].median()
df['salary'] = df['salary'].fillna(median_salary)

# Backfill minimum and maximum salary fields based on the imputed average
df['salary_min'] = df['salary_min'].fillna(df['salary'])
df['salary_max'] = df['salary_max'].fillna(df['salary'])
```

*   **Logic:** 
    1.  Cleans non-numeric currency characters from raw salary texts using regular expressions.
    2.  Computes a row-wise mean `salary` representation from minimum and maximum bounds.
    3.  Applies a median imputation to fill in missing records, preserving the row count (5,773 postings).
    4.  Ensures alignment by filling missing bounds with the imputed average.

---

## 3. Automated NLP Pipeline

An NLP pipeline is constructed to clean, tokenize, clean stopwords, and lemmatize job descriptions.

### 3.1 POS Tag Mapping Function
To enable accurate lemmatization, NLTK Perceptron tags are mapped to WordNet lexical categories (Noun, Verb, Adjective, Adverb).

```python
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatizer can read"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN) # Fallback to Noun
```

### 3.2 Text Cleaning and Preprocessing Function
Processes each job description text through step-by-step cleaning operations.

```python
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_and_lemmatize(text):
    if pd.isna(text):
        return ""
    
    # 1. Lowercase conversion
    text = str(text).lower()
    
    # 2. Strip HTML tags (e.g. <li>, <br>)
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # 3. Keep only letters and hyphens (stripping punctuation/digits)
    text = re.sub(r'[^a-zA-Z\s-]', ' ', text)
    
    # 4. Tokenization (split by whitespace)
    tokens = text.split()
    
    # 5. Stopword filtering & lemmatization
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            # Map tag and lemmatize
            pos_tag = get_wordnet_pos(token)
            lemmatized_word = lemmatizer.lemmatize(token, pos_tag)
            cleaned_tokens.append(lemmatized_word)
            
    return " ".join(cleaned_tokens)

# Apply processing to descriptions
df['clean_description'] = df['job_description'].apply(clean_and_lemmatize)

# Export cleaned dataset
os.makedirs('output', exist_ok=True)
df.to_csv(os.path.join('output', 'ai_jobs_global_2026_cleaned.csv'), index=False)
```

*   **Mechanics:**
    *   `re.sub(r'<[^>]*>', ' ', text)`: Uses regex pattern matching to remove HTML residues before processing text.
    *   `re.sub(r'[^a-zA-Z\s-]', ' ', text)`: Drops numbers and special characters to focus solely on lexical terms.
    *   `WordNetLemmatizer.lemmatize`: Normalizes tokens based on context (e.g., converting "engineered" or "engineering" back to "engineer").
    *   **Output File:** Saves the cleaned dataset containing the new `clean_description` column to `output/ai_jobs_global_2026_cleaned.csv`.

---

## 4. Feature Extraction & Statistical Modeling

Extracts numerical representations from the cleaned text using the Term Frequency-Inverse Document Frequency (TF-IDF) paradigm.

```python
# Instantiate vectorizer (Unigrams & Bigrams)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85,
    max_features=500
)

# Convert clean description to TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df['clean_description'])
feature_names = vectorizer.get_feature_names_out()

# Create dataframe representing feature weights
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
```

*   **`ngram_range=(1, 2)`:** Extracts individual terms (e.g., `python`) and key pairings (e.g., `machine learning`).
*   **DF Thresholds:** Limits noise by dropping terms appearing in fewer than 5 postings (`min_df=5`) or present in more than 85% of job descriptions (`max_df=0.85`).

---

## 5. Statistical & Visual Analysis

The final cells compute distributions and plot exploratory charts.

### 5.1 Distribution Metrics
Calculates target variable characteristics to verify normality.
```python
skewness = df['salary'].skew()
kurtosis = df['salary'].kurtosis()
print(f"Salary Skewness: {skewness:.4f}")
print(f"Salary Kurtosis: {kurtosis:.4f}")
```
*   **Result:** A high positive skewness indicates the presence of right-side salary outliers, justifying the log transformation implemented in the modeling phase.

### 5.2 Feature Correlations with Target Salary
Calculates the linear Pearson correlation coefficient between the TF-IDF feature weights of individual terms and the target variable `salary`.
```python
correlations = {}
for col in tfidf_df.columns:
    correlations[col] = tfidf_df[col].corr(df['salary'])
    
correlation_series = pd.Series(correlations).sort_values(ascending=False)
```

### 5.3 Generated Visualizations (Saved to `output/`)
The notebook generates and exports 5 figures to verify the descriptive findings:
1.  **Word Cloud (`output/wordcloud_ch4.png`):** Visualizes term frequency weight.
2.  **Top N-grams (`output/ngrams_ch4.png`):** Horizontal barplot showing the top 20 most frequent unigrams and bigrams.
3.  **Salary Boxplot (`output/salary_boxplot_ch4.png`):** Evaluates salary distribution across seniority or experience levels.
4.  **TF-IDF Correlation Heatmap (`output/tfidf_salary_heatmap_ch4.png`):** Shows correlation coefficients between selected top skills.
5.  **Top Salary Correlations (`output/tfidf_salary_top_correlations.png`):** Barplot showing terms with the strongest positive and negative linear correlations with salary (highlighting keywords like `pytorch`, `rag`, and `llm`).
