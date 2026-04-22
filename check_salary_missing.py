#!/usr/bin/env python3
import csv, json, os
path = os.path.join('output','ai_jobs_global_2026_cleaned.csv')
if not os.path.exists(path):
    print('FILE_NOT_FOUND:', path)
    raise SystemExit(1)
with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    total=0
    missing_min=0
    missing_max=0
    missing_both=0
    samples=[]
    for row in reader:
        total+=1
        minv = row.get('salary_min','') or ''
        maxv = row.get('salary_max','') or ''
        min_empty = str(minv).strip()=='' or str(minv).lower()=='nan'
        max_empty = str(maxv).strip()=='' or str(maxv).lower()=='nan'
        if min_empty:
            missing_min+=1
        if max_empty:
            missing_max+=1
        if min_empty and max_empty:
            missing_both+=1
            if len(samples)<10:
                samples.append({'job_title':row.get('job_title',''),'company':row.get('company',''),'country':row.get('country',''),'salary':row.get('salary',''),'salary_min':minv,'salary_max':maxv})
print('TOTAL_ROWS:', total)
print('MISSING_salary_min:', missing_min)
print('MISSING_salary_max:', missing_max)
print('MISSING_both:', missing_both)
print('SAMPLES_JSON:')
print(json.dumps(samples, ensure_ascii=False, indent=2))
