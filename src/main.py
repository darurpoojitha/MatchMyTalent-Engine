import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


print("\n🎯 Skill-Based Career Recommendation System\n")

# ==========================================
# Step 1: Load Dataset
# ==========================================

df = pd.read_csv("../dataset/new_jobs.csv")

print("✅ Dataset Loaded Successfully!\n")

# Strip spaces from column names (safety)
df.columns = df.columns.str.strip()

# Keep only required columns
df = df[["Job Title", "Skills", "Job Description"]]

# Drop missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates(subset=["Job Title"])
df = df.reset_index(drop=True)

print("\n📌 Total Job Titles Available:", len(df))
print("\n📌 Job Title Counts:\n")
print(df["Job Title"].value_counts().head(10))


# ==========================================
# Step 2: Clean Skills Text
# ==========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9, ]", "", text)
    return text


df["Skills"] = df["Skills"].apply(clean_text)


# ==========================================
# Step 3: TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Skills"])


# ==========================================
# Step 4: Skill Gap Function
# ==========================================

def skill_gap(user_skills, job_skills):

    user_set = set([s.strip().lower() for s in user_skills.split(",")])

    # Split job skills by comma or space
    job_set = set(re.split(",| ", job_skills.lower()))
    job_set = {s.strip() for s in job_set if s.strip() != ""}

    missing = job_set - user_set

    return list(missing)


# ==========================================
# Step 5: Recommendation Function + Score
# ==========================================

def recommend_jobs(user_skills):

    user_vec = vectorizer.transform([user_skills])

    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)[0]

    # Get top 5 jobs
    top_indices = similarity_scores.argsort()[-5:][::-1]

    recommendations = []

    for idx in top_indices:

        job_title = df.iloc[idx]["Job Title"]
        job_skills = df.iloc[idx]["Skills"]
        job_desc = df.iloc[idx]["Job Description"]

        # Matching score %
        score = round(similarity_scores[idx] * 100, 2)

        recommendations.append((job_title, job_skills, job_desc, score))

    return recommendations


# ==========================================
# Step 6: Main Program
# ==========================================

user_input = input("\nEnter your skills (comma separated): ")

results = recommend_jobs(user_input)

print("\n✅ Top Career Recommendations:\n")

for job_title, job_skills, job_desc, score in results:

    print("------------------------------------------------")
    print("📌 Job Role:", job_title)

    print("✅ Match Score:", score, "%")

    # Match level
    if score > 80:
        print("🔥 Strong Match!")
    elif score > 50:
        print("🙂 Moderate Match")
    else:
        print("⚠️ Low Match - Need More Skills")

    # Skill Gap
    gaps = skill_gap(user_input, job_skills)
    print("⚠️ Missing Skills:", gaps[:6] if gaps else "None 🎉")

    # Description Preview
    print("\n📝 Job Description Preview:")
    print(job_desc[:200], "...")

    print("------------------------------------------------\n")
