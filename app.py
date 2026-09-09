import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Load saved files
# =========================

tfidf = joblib.load("tfidf_vectorizer.pkl")

tfidf_matrix = joblib.load("tfidf_matrix.pkl")

df = pd.read_pickle("resume_data.pkl")


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)


st.title("🤖 AI Resume Screening System")

st.write(
    "Enter a job description and the system will rank resumes based on similarity."
)


# =========================
# Job Description Input
# =========================

job_description = st.text_area(
    "Enter Job Description",
    height=200
)


# =========================
# Analyze Button
# =========================

if st.button("Analyze Resumes"):

    if job_description.strip() == "":
        
        st.warning(
            "Please enter a Job Description first."
        )

    else:

        # Convert job description to TF-IDF vector
        job_vector = tfidf.transform(
            [job_description]
        )


        # Calculate similarity scores
        scores = cosine_similarity(
            job_vector,
            tfidf_matrix
        )[0]


        # Copy dataframe to avoid overwriting
        results = df.copy()


        # Add scores
        results["Match_Score"] = scores * 100


        # Sort results
        results = results.sort_values(
            by="Match_Score",
            ascending=False
        )


        # =========================
        # Top Candidates
        # =========================

        st.subheader(
            "🏆 Top Matching Candidates"
        )


        top_candidates = results[
            [
                "ID",
                "Category",
                "Match_Score"
            ]
        ].head(10)


        st.dataframe(
            top_candidates,
            use_container_width=True
        )


        # =========================
        # Shortlist
        # =========================

        st.subheader(
            "✅ Shortlisted Candidates"
        )


        threshold = st.slider(
            "Select Shortlist Threshold (%)",
            min_value=0,
            max_value=100,
            value=30
        )


        shortlisted = results[
            results["Match_Score"] >= threshold
        ]


        st.write(
            f"Number of shortlisted candidates: {len(shortlisted)}"
        )


        st.dataframe(
            shortlisted[
                [
                    "ID",
                    "Category",
                    "Match_Score"
                ]
            ],
            use_container_width=True
        )


        # =========================
        # Download Results
        # =========================

        csv = shortlisted[
            [
                "ID",
                "Category",
                "Match_Score"
            ]
        ].to_csv(
            index=False
        )


        st.download_button(
            label="📥 Download Shortlist CSV",
            data=csv,
            file_name="shortlisted_candidates.csv",
            mime="text/csv"
        )