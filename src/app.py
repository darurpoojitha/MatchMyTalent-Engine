from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load Dataset
# -------------------------------
print("✅ Loading Dataset...")

df = pd.read_csv("../dataset/new_jobs.csv")
df = df.dropna()

print("✅ Dataset Loaded Successfully!")
print("Columns:", df.columns)

# -------------------------------
# TF-IDF Vectorizer
# -------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Skills"])

# -------------------------------
# Course Suggestions Dictionary
# -------------------------------
course_map = {
    "MACHINE LEARNING ENGINEER": [
        "Deep Learning Specialization – Coursera",
        "Machine Learning by Andrew Ng – Coursera",
        "ML Model Deployment with Flask – Udemy"
    ],

    "ARTIFICIAL INTELLIGENCE ENGINEER": [
        "AI For Everyone – Coursera",
        "Neural Networks & Deep Learning – Coursera",
        "AI Foundations – Udacity"
    ],

    "DATA SCIENTIST": [
        "Data Science Bootcamp – Udemy",
        "Python for Data Science – Kaggle",
        "Statistics for Data Science – Coursera"
    ],

    "NATURAL LANGUAGE PROCESSING ENGINEER": [
        "NLP Specialization – Coursera",
        "Transformers with HuggingFace – Udemy",
        "Text Analytics – edX"
    ],

    "DEVOPS ENGINEER": [
        "Docker & Kubernetes – Udemy",
        "AWS DevOps Certification – Coursera",
        "CI/CD Pipelines – edX"
    ]
}

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_jobs(user_skills):

    # Convert user skills into list
    user_skill_list = [s.strip().lower() for s in user_skills.split(",")]

    # Vectorize user input
    user_vec = vectorizer.transform([user_skills])

    # Similarity Score
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)

    # Top 5 recommendations
    top_indices = similarity_scores[0].argsort()[-5:][::-1]

    recommendations = []

    for idx in top_indices:
        row = df.iloc[idx]

        # Job Title (Correct Column)
        job_title = row["Job Title"]

        # Job Description
        description = row["Job Description"]

        # Match Score %
        match_score = round(similarity_scores[0][idx] * 100, 2)

        # Job Skills List
        job_skill_list = [s.strip().lower() for s in row["Skills"].split(",")]

        # Missing Skills
        missing_skills = [
            skill for skill in job_skill_list
            if skill not in user_skill_list
        ]

        # Suggested Courses
        course_list = course_map.get(
            job_title.upper(),
            ["No course suggestions available"]
        )

        # Append Result
        recommendations.append({
            "job": job_title,
            "score": match_score,
            "missing": missing_skills,
            "desc": description[:200] + "...",
            "courses": course_list
        })

    return recommendations

# -------------------------------
# Home Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = None

    if request.method == "POST":
        skills = request.form.get("skills")

        if skills:
            recommendations = recommend_jobs(skills)

    return render_template("index.html", recommendations=recommendations)

# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
