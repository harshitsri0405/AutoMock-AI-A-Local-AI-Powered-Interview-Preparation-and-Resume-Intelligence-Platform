import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. PDF se text extract karna
def extract_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text.lower()

# 2. ATS Score nikalna
def get_ats_score(resume_text, job_description):
    text_list = [resume_text, job_description]
    cv = TfidfVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(text_list)
    score = cosine_similarity(count_matrix)[0][1] * 100
    return round(score, 2)

# 3. Dataset se skills match karna
def detect_skills_and_questions(resume_text, dataset_path):
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip().str.lower() # Column names safe clean
    available_skills = df['skill'].dropna().unique() 
    
    detected_skills = []
    for skill in available_skills:
        if str(skill).lower().strip() in resume_text:
            detected_skills.append(skill)
            
    return detected_skills