#!/bin/sh
for file in ../../data/proc/raw_csv/*.csv; do
    echo "$file"
    python vectorizer_transform.py --vectorizer_path ../../data/proc/vectorizer_1/vectorizer.joblib --input $file --output_folder ../../data/proc/vectorizer_1/ &
    sleep 300
done