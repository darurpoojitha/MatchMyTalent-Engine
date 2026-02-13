from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Flask App
app = Flask(__name__)

# ============================
# Load Dataset
# ============================
df = pd.read_csv("../dataset/new_jobs.csv")
df = df.dropna()

# ============================
# TF-IDF Vectorizer (Skills Column)
# ============================
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Skills"])

# ============================
# Course Suggestions Map
# ============================
course_map = {
    "Artificial Intelligence": [
        "AI For Everyone – Coursera",
        "Neural Networks & Deep Learning – Coursera",
        "AI Foundations – Udacity"
    ],
    "Machine Learning": [
        "Machine Learning by Andrew Ng – Coursera",
        "Deep Learning Specialization – Coursera",
        "ML Ops Basics – Udemy"
    ],
    "Data Scientist": [
        "Data Science Professional Certificate – IBM",
        "Python for Data Science – Coursera",
        "Statistics for DS – Khan Academy"
    ],
    "Web Developer": [
        "Full Stack Web Dev – Udemy",
        "HTML CSS JavaScript – Coursera",
        "React Bootcamp – Scrimba"
    ]
}

# ============================
# Roadmap Generator
# ============================
def generate_roadmap(missing_skills):
    roadmap = []
    for skill in missing_skills[:5]:
        roadmap.append(f"Learn {skill}")
    return roadmap

# ============================
# Recommendation Function
# ============================
def recommend_jobs(user_input, mode="skills"):

    recommendations = []

    # ----------------------------
    # MODE 1: Search by Job Title
    # ----------------------------
    if mode == "job":

        matched = df[df["Job Title"].str.lower().str.contains(user_input.lower())]

        if matched.empty:
            return [{
                "job": "No Job Found",
                "score": 0,
                "required": [],
                "missing": [],
                "roadmap": [],
                "desc": "Try another job title.",
                "courses": [],
                "search_type": "job"
            }]

        for _, row in matched.head(3).iterrows():
            job_role = row["Job Title"]
            required_skills = [s.strip() for s in row["Skills"].split(",")]

            recommendations.append({
                "job": job_role,
                "score": 100,
                "required": required_skills,
                "missing": [],
                "roadmap": [],
                "desc": row["Job Description"],
                "courses": [],
                "search_type": "job"
            })

        return recommendations

    # ----------------------------
    # MODE 2: Search by Skills
    # ----------------------------
    user_skill_list = [s.strip().lower() for s in user_input.split(",")]

    # Vectorize user input
    user_vec = vectorizer.transform([user_input])

    # Similarity scores
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)

    # Top 5 matches
    top_indices = similarity_scores[0].argsort()[-5:][::-1]

    for idx in top_indices:
        row = df.iloc[idx]

        job_role = row["Job Title"]
        description = row["Job Description"]

        job_skill_list = [s.strip().lower() for s in row["Skills"].split(",")]

        # Missing Skills
        missing_skills = [
            skill for skill in job_skill_list if skill not in user_skill_list
        ]

        # Match Score
        match_score = round(similarity_scores[0][idx] * 100, 2)

        # Roadmap
        roadmap = generate_roadmap(missing_skills)

        # Course Suggestion
        course_list = ["No course suggestions available"]
        for key in course_map:
            if key.lower() in job_role.lower():
                course_list = course_map[key]
                break

        recommendations.append({
            "job": job_role,
            "score": match_score,
            "required": job_skill_list,
            "missing": missing_skills,
            "roadmap": roadmap,
            "desc": description,
            "courses": course_list,
            "search_type": "skills"
        })

    return recommendations

# ============================
# Home Route
# ============================
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    mode = "skills"  # default

    if request.method == "POST":
        user_input = request.form.get("skills")
        mode = request.form.get("mode")  # get selected mode
        if user_input:
            recommendations = recommend_jobs(user_input, mode)

    return render_template("index.html", recommendations=recommendations, mode=mode)

# ============================
# Run App
# ============================
if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# Flask App
# ============================
app = Flask(__name__)

# ============================
# Load Dataset
# ============================
df = pd.read_csv("../dataset/new_jobs.csv")
df = df.dropna()

# ============================
# Skill Alias Normalization
# ============================
skill_aliases = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nn": "neural networks",
    "nlp": "natural language processing"
}

def normalize_skills(skill_list):
    normalized = []
    for skill in skill_list:
        skill = skill.strip().lower()
        if skill in skill_aliases:
            normalized.append(skill_aliases[skill])
        else:
            normalized.append(skill)
    return normalized

# ============================
# TF-IDF Vectorizer
# ============================
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Skills"])

# ============================
# Course Suggestions
# ============================
course_map = {
    "artificial intelligence": [
        "AI For Everyone – Coursera",
        "Neural Networks & Deep Learning – Coursera",
        "AI Foundations – Udacity"
    ],
    "machine learning": [
        "Machine Learning by Andrew Ng – Coursera",
        "Deep Learning Specialization – Coursera",
        "ML Ops Basics – Udemy"
    ],
    "data scientist": [
        "Data Science Professional Certificate – IBM",
        "Python for Data Science – Coursera",
        "Statistics for DS – Khan Academy"
    ],
    "web developer": [
        "Full Stack Web Dev – Udemy",
        "HTML CSS JavaScript – Coursera",
        "React Bootcamp – Scrimba"
    ]
}

# ============================
# Roadmap Generator
# ============================
def generate_roadmap(missing_skills):
    roadmap = []
    for skill in missing_skills[:5]:
        roadmap.append(f"Learn {skill}")
    return roadmap

# ============================
# Recommendation Function
# ============================
def recommend_jobs(user_input, mode="skills"):

    recommendations = []

    # ----------------------------
    # MODE 1: Search by Job Title
    # ----------------------------
    if mode == "job":

        matched = df[df["Job Title"].str.lower().str.contains(user_input.lower())]

        if matched.empty:
            return [{
                "job": "No Job Found",
                "score": 0,
                "required": [],
                "matched": [],
                "missing": [],
                "roadmap": [],
                "desc": "Try another job title.",
                "courses": [],
                "explanation": "No matching job titles found.",
                "search_type": "job"
            }]

        for _, row in matched.head(3).iterrows():
            job_role = row["Job Title"]
            required_skills = [s.strip().lower() for s in row["Skills"].split(",")]

            recommendations.append({
                "job": job_role,
                "score": 100,
                "required": required_skills,
                "matched": required_skills,
                "missing": [],
                "roadmap": [],
                "desc": row["Job Description"],
                "courses": [],
                "explanation": "Exact job title match.",
                "search_type": "job"
            })

        return recommendations

    # ----------------------------
    # MODE 2: Search by Skills
    # ----------------------------

    user_skill_list = normalize_skills(user_input.split(","))

    # Vectorize user input
    user_vec = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)

    # Get sorted indices
    sorted_indices = similarity_scores[0].argsort()[::-1]

    for idx in sorted_indices[:10]:  # Check top 10, filter later
        row = df.iloc[idx]

        job_role = row["Job Title"]
        description = row["Job Description"]

        job_skill_list = normalize_skills(row["Skills"].split(","))

        # Skill overlap
        matched_skills = list(set(user_skill_list) & set(job_skill_list))
        missing_skills = list(set(job_skill_list) - set(user_skill_list))

        overlap_score = len(matched_skills) / len(job_skill_list) if job_skill_list else 0

        # Hybrid scoring
        final_score = (0.7 * similarity_scores[0][idx]) + (0.3 * overlap_score)
        match_score = round(final_score * 100, 2)

        if match_score < 20:
            continue

        roadmap = generate_roadmap(missing_skills)

        # Course suggestions
        course_list = ["No course suggestions available"]
        for key in course_map:
            if key in job_role.lower():
                course_list = course_map[key]
                break

        recommendations.append({
            "job": job_role,
            "score": match_score,
            "required": job_skill_list,
            "matched": matched_skills,
            "missing": missing_skills,
            "roadmap": roadmap,
            "desc": description,
            "courses": course_list,
            "explanation": f"Matched {len(matched_skills)} out of {len(job_skill_list)} required skills.",
            "search_type": "skills"
        })

        if len(recommendations) == 5:
            break

    return recommendations

# ============================
# Home Route
# ============================
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    mode = "skills"

    if request.method == "POST":
        user_input = request.form.get("skills")
        mode = request.form.get("mode")

        if user_input:
            recommendations = recommend_jobs(user_input, mode)

    return render_template("index.html", recommendations=recommendations, mode=mode)

# ============================
# Run App
# ============================
if __name__ == "__main__":
    app.run(debug=True)
