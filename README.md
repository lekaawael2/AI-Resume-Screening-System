# AI Resume Screening System

## Project Overview

An AI-based Resume Screening System that automatically analyzes resumes and ranks candidates based on their similarity to a given job description using Natural Language Processing (NLP).

The system helps recruiters save time by automatically filtering and shortlisting suitable candidates.

---

## Features

- Resume text preprocessing
- NLP cleaning pipeline
- Stopwords removal
- Tokenization
- Lemmatization
- TF-IDF feature extraction
- Cosine Similarity matching
- Resume ranking system
- Candidate shortlisting
- Streamlit web application

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit

---

## Dataset

Resume dataset containing:
- Resume text
- Resume category

Total resumes:
2484

---

## How It Works

1. User enters a job description.
2. The system converts the job description into TF-IDF vectors.
3. Cosine similarity is calculated between the job description and resumes.
4. Resumes are ranked according to similarity score.
5. Top candidates are displayed and shortlisted.

---

## Run Locally

Install requirements:
