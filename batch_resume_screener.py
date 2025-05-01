import streamlit as st
import fitz
import docx2txt
import os
from sentence_transformers import SentenceTransformer, util
from tempfile import NamedTemporaryFile
import shutil
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join([page.get_text() for page in doc])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    else:
        return ""

def save_file(file):
    suffix = ".pdf" if file.type == "application/pdf" else ".docx"
    with NamedTemporaryFile(delete=False, suffix=suffix, dir="temp_resumes") as tmp:
        tmp.write(file.read())
        return tmp.name

def rank_resumes(job_desc, resumes):
    jd_embedding = model.encode(job_desc, convert_to_tensor=True)
    results = []

    for file, text, path in resumes:
        if not text.strip():
            continue
        res_embedding = model.encode(text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(res_embedding, jd_embedding).item() * 100
        results.append((file.name, score, path))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]

st.set_page_config(page_title="Batch Resume Screener", layout="wide")
st.title("Batch Resume Screening with AI")

uploaded_files = st.file_uploader("Upload Multiple Resumes", type=["pdf", "docx"], accept_multiple_files=True)
job_description = st.text_area("Paste the Job Description")

if uploaded_files and job_description:
    if not os.path.exists("temp_resumes"):
        os.mkdir("temp_resumes")

    with st.spinner("Processing resumes..."):
        resume_data = []
        for file in uploaded_files:
            file.seek(0)
            resume_text = extract_text(file)
            file.seek(0)
            saved_path = save_file(file)
            resume_data.append((file, resume_text, saved_path))

        top_matches = rank_resumes(job_description, resume_data)

    st.subheader("🏆 Top 5 Resume Matches")
    if top_matches:
        df = pd.DataFrame(
            [{
                "Name": name,
                "Score (%)": f"{score:.2f}",
                "Download": f"[⬇ Download]({path})"
            } for name, score, path in top_matches]
        )
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)

        for name, score, path in top_matches:
            with open(path, "rb") as f:
                st.download_button(f"Download {name}", f, file_name=name)

    if resume_data:
        all_text = " ".join([text for _, text, _ in resume_data])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

        st.subheader("📊 WordCloud of Resume Content")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    if st.button("Clear Temp Files"):
        shutil.rmtree("temp_resumes")
        st.success("Cleared!")
