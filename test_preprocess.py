import pandas as pd
import re
import sys
import os
from nltk.stem import WordNetLemmatizer
import nltk

# ensure venv python is used
csv_path = 'ai_jobs_global_2026.csv'
if not os.path.exists(csv_path):
    print('ERROR: CSV not found at', csv_path)
    sys.exit(2)

# download minimal NLTK resources needed (no punkt)
for pkg in ['averaged_perceptron_tagger','wordnet','omw-1.4','stopwords']:
    try:
        if pkg == 'averaged_perceptron_tagger':
            nltk.data.find('taggers/averaged_perceptron_tagger')
        elif pkg == 'wordnet':
            nltk.data.find('corpora/wordnet')
        elif pkg == 'omw-1.4':
            nltk.data.find('corpora/omw-1.4')
        elif pkg == 'stopwords':
            nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download(pkg)

from nltk.corpus import stopwords, wordnet

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def nltk_pos_to_wordnet(nltk_pos):
    if nltk_pos.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_text(text):
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # regex tokenizer (avoid punkt)
    tokens = re.findall(r"\b[a-z]+\b", text)
    # POS tagging requires tokens; fallback: if empty, return ''
    if not tokens:
        return ''
    pos_tags = nltk.pos_tag(tokens)
    cleaned = []
    for token, pos in pos_tags:
        if token in stop_words or not token.isalpha():
            continue
        wn_pos = nltk_pos_to_wordnet(pos)
        lemma = lemmatizer.lemmatize(token, wn_pos)
        cleaned.append(lemma)
    return ' '.join(cleaned)


def build_salary(df):
    # create unified salary
    if 'salary' not in df.columns:
        if 'salary_min' in df.columns or 'salary_max' in df.columns:
            if 'salary_min' in df.columns:
                df['salary_min'] = pd.to_numeric(df['salary_min'], errors='coerce')
            if 'salary_max' in df.columns:
                df['salary_max'] = pd.to_numeric(df['salary_max'], errors='coerce')
            if 'salary_min' in df.columns and 'salary_max' in df.columns:
                df['salary'] = df[['salary_min', 'salary_max']].mean(axis=1)
            elif 'salary_min' in df.columns:
                df['salary'] = df['salary_min']
            else:
                df['salary'] = df['salary_max']
    if 'salary' in df.columns:
        med = pd.to_numeric(df['salary'], errors='coerce').median()
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(med)
    return df


print('Loading CSV...')
df = pd.read_csv(csv_path)
print('Shape:', df.shape)
print('Columns:', list(df.columns))

# build salary
df = build_salary(df)
print('\nSalary present?', 'salary' in df.columns)
if 'salary' in df.columns:
    print(df['salary'].describe())

# required_skills handling
if 'required_skills' in df.columns:
    before = df['required_skills'].isnull().sum()
    df['required_skills'] = df['required_skills'].fillna('To be extracted')
    after = df['required_skills'].isnull().sum()
    print('\nrequired_skills filled:', before - after)
else:
    print('\nrequired_skills column missing')

# apply preprocessing to first 20 rows only for speed
if 'job_description' in df.columns:
    sample = df['job_description'].fillna('').astype(str).head(20).tolist()
    processed = [preprocess_text(t) for t in sample]
    for i,(orig,proc) in enumerate(zip(sample, processed)):
        print('\n--- ROW', i, '---')
        print('ORIG:', orig[:200])
        print('CLEAN:', proc[:200])
    print('\nPreprocessing test completed successfully')
else:
    print('\njob_description column missing')

print('\nAll tests done')
