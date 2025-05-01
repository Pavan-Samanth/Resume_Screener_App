# Resume Screener App

This Streamlit app allows you to upload multiple resumes, compare them with a job description using semantic similarity, and ranks the top 5 matches. It also generates a word cloud from the resume content.

## Features

- Upload multiple resumes (PDF/DOCX)
- Semantic similarity scoring using `sentence-transformers`
- Top 5 matching resumes with score table and download buttons
- WordCloud from all resumes
- Docker support
- Streamlit Cloud ready

## Run Locally

```bash
pip install -r requirements.txt
streamlit run batch_resume_screener.py
```

## Docker

```bash
docker build -t resume-screener .
docker run -p 8501:8501 resume-screener
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud
3. Select your repo and deploy.