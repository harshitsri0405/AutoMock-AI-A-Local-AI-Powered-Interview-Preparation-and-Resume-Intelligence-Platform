import streamlit as st
import pandas as pd
import os
import pickle
import re
import time
from google import genai
from core_logic import extract_text, detect_skills_and_questions
from dotenv import load_dotenv

# ==============================================================================
# 🎯 SYSTEM STRATIFICATION: ABSOLUTE FIXED PATH INITIALIZATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_file_path = os.path.join(BASE_DIR, ".env")
backup_env_path = os.path.join(BASE_DIR, ".env.txt")

# Strict dynamic scan mapping to enforce true variable loading
env_loaded = False
if os.path.exists(env_file_path):
    env_loaded = load_dotenv(dotenv_path=env_file_path, override=True)
elif os.path.exists(backup_env_path):
    env_loaded = load_dotenv(dotenv_path=backup_env_path, override=True)
else:
    load_dotenv()

st.set_page_config(page_title="AutoMock AI", layout="wide", page_icon="🚀")

# ==============================================================================
# CUSTOM GLASSMORPHISM & NEON UI STYLING (CSS INJECTION)
# ==============================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #38bdf8;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.25);
        border-color: #0ea5e9;
    }
    .card-title-blue { color: #38bdf8 !important; font-weight: 700 !important; margin-bottom: 5px; }
    .card-title-purple { color: #c084fc !important; font-weight: 700 !important; margin-bottom: 5px; }
    .card-title-green { color: #4ade80 !important; font-weight: 700 !important; margin-bottom: 5px; }
    .card-title-orange { color: #fb923c !important; font-weight: 700 !important; margin-bottom: 5px; }
    .card-desc { color: #94a3b8; font-size: 14px; margin-bottom: 15px; }
    .neon-hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(56, 189, 248, 0), rgba(56, 189, 248, 0.75), rgba(56, 189, 248, 0));
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- DIRECTORY MAPPING BASED ON YOUR SCREENSHOT ---
TRAIN_DATASET_PATH = os.path.join(BASE_DIR, "dataset_part1.csv")
TEST_DATASET_PATH = os.path.join(BASE_DIR, "dataset_part2.csv")

MODEL_PATH = os.path.join(BASE_DIR, "role_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Split Data Integrity Validation Checks
if not os.path.exists(TRAIN_DATASET_PATH) or not os.path.exists(TEST_DATASET_PATH):
    st.error("❌ Structural Exception: 'dataset_part1.csv' (Train) or 'dataset_part2.csv' (Test) split matrix missing at project folder.")
    st.stop()
else:
    # Train Matrix Mapping (For Dropdowns, Blueprints & Baseline Rules)
    df_train = pd.read_csv(TRAIN_DATASET_PATH)
    df_train.columns = df_train.columns.str.strip().str.lower()
    
    # Test Matrix Mapping (For Mock Evaluations & Question Isolation)
    df_test = pd.read_csv(TEST_DATASET_PATH)
    df_test.columns = df_test.columns.str.strip().str.lower()

# ==============================================================================
# SECURE PRODUCTION ENVIRONMENT & AUTH REGISTRATION (NO HARDCODING)
# ==============================================================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

client = None
if GEMINI_KEY and GEMINI_KEY.strip():
    try:
        client = genai.Client(api_key=GEMINI_KEY.strip())
    except Exception as e:
        st.error(f"Critical: Gemini Client Initialization Failed: {e}")
else:
    st.warning("⚠️ Environment Config: API Key not found in Environment. Please review your Streamlit Secrets / Local configuration.")

# ==============================================================================
# INDIVIDUAL LIGHTWEIGHT LIVE INTERVIEW RESPONDER (SHORT & CRISP)
# ==============================================================================
def get_ai_coaching_feedback(question, standard_concept):
    if not client:
        return f"• {standard_concept}\n\n*(Note: System running in Offline Fallback Mode)*"
    
    prompt = f"""
    You are an elite technical interviewer. Provide an extremely short, interview-ready model answer for this question.
    Give exactly 2 or 3 short bullet points. Each bullet point must be crisp and concise enough that a candidate can comfortably speak it in front of an interviewer within 20 seconds.
    Do not include any introductory greetings, evaluations, scores, or meta text. Start directly with the first bullet point.

    Technical Question: {question}
    Concept Hint: {standard_concept}
    """
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        if response and response.text and len(response.text.strip()) > 5:
            return response.text.strip()
        return f"• {standard_concept}"
    except Exception as e:
        return f"❌ Live Generation Error: {str(e)} \n\n• **Dataset Solution:** {standard_concept}"

def get_resume_improvement_tips(resume_text, predicted_role):
    if not client:
        return "⚠️ Gemini API client not initialized. Cannot run active analysis profile."
    prompt = f"Review this resume text for ATS parameters based on target role '{predicted_role}':\n{resume_text}"
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"❌ Error executing LLM audit: {str(e)}"

# ==============================================================================
# ML METRICS PROFILE CLASSIFIER MATRIX
# ==============================================================================
def predict_role_from_resume(resume_text):
    text_lower = resume_text.lower()
    keyword_rules = {
        "data scientist": ["python", "machine learning", "data science", "pandas", "scikit-learn"],
        "generative ai engineer": ["llm", "rag", "langchain", "prompt engineering", "gemini", "openai"],
        "cloud infrastructure planner": ["aws", "cloud", "devops", "docker", "kubernetes", "s3"],
        "backend engineer": ["java", "spring boot", "node.js", "express", "sql", "mongodb"],
        "frontend web developer": ["react", "javascript", "html", "css", "next.js", "angular"],
        "c developer": ["c++", "embedded", "pointers", "data structures", "system verilog"]
    }
    
    for role, tokens in keyword_rules.items():
        if any(token in text_lower for token in tokens):
            return role
            
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        try:
            with open(MODEL_PATH, "rb") as model_file, open(VEC_PATH, "rb") as vec_file:
                model = pickle.load(model_file)
                vectorizer = pickle.load(vec_file)
            text_vectorized = vectorizer.transform([resume_text])
            ml_pred = str(model.predict(text_vectorized)[0]).strip()
            return ml_pred if ml_pred else "C Developer"
        except Exception:
            return "C Developer"
            
    return "General Software Engineer"

# --- STATE MANAGEMENT CONFIGURATION ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "home"
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "ai_feedback_cache" not in st.session_state:  
    st.session_state.ai_feedback_cache = {}
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = {}
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = ""
if "cached_resume_text" not in st.session_state:
    st.session_state.cached_resume_text = ""
if "timer_enabled" not in st.session_state:
    st.session_state.timer_enabled = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# ==============================================================================
# SIDEBAR CENTRAL DECK CONTROLS
# ==============================================================================
st.sidebar.title("Control Center")
st.sidebar.subheader("Preferences")
st.session_state.timer_enabled = st.sidebar.toggle("Enable Live Timer", value=st.session_state.timer_enabled)

st.sidebar.markdown("---")
st.sidebar.subheader("Additional Tools")

if st.sidebar.button("Resume Architect", use_container_width=True):
    st.session_state.app_mode = "resume_builder_screen"
    st.rerun()
    
if st.session_state.app_mode != "home":
    if st.sidebar.button("Central Dashboard", use_container_width=True, type="primary"):
        st.session_state.app_mode = "home"
        st.rerun()

# ==============================================================================
# SCREEN 1: CENTRAL CONTROL DASHBOARD
# ==============================================================================
if st.session_state.app_mode == "home":
    st.markdown("<h1 style='text-align: center; color: #38bdf8; font-weight: 800;'>AUTOMOCK AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 16px;'>A Local AI-Powered Interview Preparation and Resume Analysis Platform</p>", unsafe_allow_html=True)
    st.markdown("<div class='neon-hr'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='feature-card'>
        <h3 class='card-title-blue'>Core Domain Simulator</h3>
        <p class='card-desc'>Run customized mock question sets extracted dynamically from the local dataset matrix records.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Skills mapping generated through Training Split Dataset
    if 'skill' in df_train.columns:
        unique_topics = sorted(df_train['skill'].dropna().unique())
        unique_topics_display = [str(t).upper() for t in unique_topics]
        selected_topic_display = st.selectbox("Choose Technical Domain Target:", unique_topics_display, key="main_deck_sb")
        interview_topic = selected_topic_display.lower().strip()
    
    if st.button("Launch Technical Test ⚡", use_container_width=True):
        if interview_topic != "":
            # 🎯 STAGE-TEST DESIGNATION: Questions are sampled strictly from unseen test data matrix records
            all_topic_qs = df_test[df_test['skill'].str.lower().str.strip() == interview_topic]
            if all_topic_qs.empty:
                all_topic_qs = df_test[df_test['skill'].str.contains(interview_topic, na=False)]
            
            if not all_topic_qs.empty:
                filtered_qs = all_topic_qs.sample(n=min(5, len(all_topic_qs)))
                questions_package = []
                for _, row in filtered_qs.iterrows():
                    questions_package.append({
                        "question": row.get('question', 'Question Missing.'),
                        "skill": row.get('skill', 'General'),
                        "answer": row.get('answer', 'Answer Missing.')
                    })
                st.session_state.questions = questions_package
                st.session_state.selected_topic = interview_topic.upper()
                st.session_state.answers = {}
                st.session_state.ai_feedback_cache = {} 
                st.session_state.current_question_index = 0
                st.session_state.start_time = time.time()
                st.session_state.app_mode = "interview_topic"
                st.rerun()

    st.markdown("<div class='neon-hr'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #e2e8f0; margin-bottom: 15px;'>Adaptive Candidate Profile Solutions</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop your engineering CV / Profile Sheet here (PDF only):", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Extracting profile metrics..."):
            st.session_state.cached_resume_text = extract_text(uploaded_file)
            st.session_state.resume_analysis["predicted_role"] = predict_role_from_resume(st.session_state.cached_resume_text)
        
        st.success(f"🎯 Local ML Prediction Alignment Model: {st.session_state.resume_analysis['predicted_role'].upper()}")
        
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown("""
            <div class='feature-card'>
                <h3 class='card-title-green'>CV-to-Question Engine</h3>
                <p class='card-desc'>Maps your resume content automatically with matching dataset questions for personalized evaluation.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Initiate Adaptive Interview 🎯", use_container_width=True, type="primary"):
                # 🎯 ADAPTIVE MATRIX: Skill matching and selection maps strictly using TEST split metadata
                detected_skills = detect_skills_and_questions(st.session_state.cached_resume_text, TEST_DATASET_PATH)
                all_resume_qs = df_test[df_test['skill'].isin(detected_skills)] if detected_skills else pd.DataFrame()
                if all_resume_qs.empty:
                    all_resume_qs = df_test[df_test['question_type'] == 'behavioral'] if 'question_type' in df_test.columns else df_test
                
                filtered_qs = all_resume_qs.sample(n=min(5, len(all_resume_qs)))
                questions_package = []
                for _, row in filtered_qs.iterrows():
                    questions_package.append({
                        "question": row.get('question', 'Question Missing.'),
                        "skill": row.get('skill', 'General'),
                        "answer": row.get('answer', 'Answer Details Empty.')
                    })
                st.session_state.questions = questions_package
                st.session_state.answers = {}
                st.session_state.ai_feedback_cache = {} 
                st.session_state.current_question_index = 0
                st.session_state.start_time = time.time()
                st.session_state.app_mode = "interview_resume"
                st.rerun()

        with row2_col2:
            st.markdown("""
            <div class='feature-card'>
                <h3 class='card-title-orange'>ATS Resume Analyzer</h3>
                <p class='card-desc'>Provides comprehensive market alignment feedback and strategic missing-keyword metrics via LLM layers.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Generate Premium Audit Feedback 🧠", use_container_width=True):
                st.session_state.app_mode = "resume_reviewer_screen"
                st.rerun()

# ==============================================================================
# SCREEN 2: ATS RESUME ANALYZER WORKFLOW
# ==============================================================================
elif st.session_state.app_mode == "resume_reviewer_screen":
    st.subheader("ATS Resume Analyzer Panel")
    role_prediction = st.session_state.resume_analysis.get("predicted_role", "Unknown Profile")
    st.metric(label="Best Suited Job Profile (Predicted via Local ML)", value=role_prediction.upper())
    
    with st.spinner("Gemini LLM layer executing automated structural audit..."):
        audit_report = get_resume_improvement_tips(st.session_state.cached_resume_text, role_prediction)
        st.markdown(audit_report)
    if st.button("⬅️ Back to Central Hub"):
        st.session_state.app_mode = "home"
        st.rerun()

# ==============================================================================
# SCREEN 3: RESUME ARCHITECT SYSTEM BUILDER
# ==============================================================================
elif st.session_state.app_mode == "resume_builder_screen":
    st.markdown("<h2 style='color: #c084fc;'>Resume Architect</h2>", unsafe_allow_html=True)
    st.write("Compile an industry-standard structured professional resume profile driven by authenticated local matrix metrics.")
    st.markdown("---")
    
    with st.form("resume_builder_form"):
        st.markdown("### 👤 Contact Information")
        c1, c2 = st.columns(2)
        with c1:
            u_name = st.text_input("Full Name:", placeholder="Anmol Kumar")
            u_email = st.text_input("Email Address:", placeholder="anmol@example.com")
        with c2:
            u_phone = st.text_input("Phone Number:", placeholder="+91 9876543210")
            u_links = st.text_input("Professional Links (LinkedIn / GitHub):", placeholder="linkedin.com/in/username")
            
        st.markdown("### 🎓 Education")
        ed_col1, ed_col2 = st.columns(2)
        with ed_col1:
            u_edu = st.text_input("Degree & Major:", placeholder="B.Tech in Computer Science")
            u_univ = st.text_input("University / Institute Name:", placeholder="Your University Name")
        with ed_col2:
            u_edu_year = st.text_input("Graduation Timeline:", placeholder="e.g., 2022 - 2026")
            u_edu_metrics = st.text_input("Academic Performance (CGPA / Percentage):", placeholder="e.g., 8.5 CGPA")

        st.markdown("### 🛠️ Technical Skill Matrix")
        target_profile = st.text_input("Target Job Profile / Role Alignment:", placeholder="e.g., Software Engineer")
        
        if 'skill' in df_train.columns:
            all_skills_list = sorted([str(s).upper() for s in df_train['skill'].dropna().unique()])
            selected_skills_list = st.multiselect("Select Core Technical Domain Proficiencies:", all_skills_list)
        else:
            selected_skills_list = []
            
        u_tools_frameworks = st.text_input("Developer Tools & Frameworks:", placeholder="Git, GitHub, VS Code, Docker, AWS")

        st.markdown("### 💼 Professional Work Experience")
        exp_enabled = st.checkbox("Add Work Experience / Internships?", value=True)
        if exp_enabled:
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                u_job_title = st.text_input("Job Title / Internship Role:", placeholder="e.g., Software Engineering Intern")
                u_company = st.text_input("Company / Organization:", placeholder="e.g., Tech Solutions")
            with exp_col2:
                u_job_duration = st.text_input("Employment Duration:", placeholder="e.g., June 2025 - Present")
                u_job_desc = st.text_area("Key Responsibilities:", placeholder="Optimized database execution metrics, handled system components pipelines.")
        
        st.markdown("### 🚀 Featured Engineering Projects")
        proj_col1, proj_col2 = st.columns(2)
        with proj_col1:
            u_proj_title = st.text_input("Project Name:", placeholder="e.g., AutoMock AI Hub")
            u_proj_tech = st.text_input("Technologies Used:", placeholder="e.g., Python, Streamlit, Gemini API")
        with proj_col2:
            u_proj_desc = st.text_area("Comprehensive Project Description:", placeholder="Built a secure evaluation interface processing parameters via localized LLM validation algorithms.")

        submit_build = st.form_submit_button("Compile Executive Professional Resume ⚡", use_container_width=True)
        
    if submit_build:
        if u_name == "" or target_profile == "":
            st.error("Name and Target Profile fields cannot be left empty!")
        else:
            lower_selected_skills = [s.lower().strip() for s in selected_skills_list]
            matched_rows = df_train[df_train['skill'].isin(lower_selected_skills)]
            dataset_project_bullets = []
            
            if not matched_rows.empty:
                grouped = matched_rows.groupby('skill').head(1)
                for idx, row in grouped.iterrows():
                    sk = str(row['skill']).upper()
                    ans_concept = str(row['answer'])
                    dataset_project_bullets.append(f"- **Advanced Domain Implementation ({sk}):** Developed enterprise logic criteria: {ans_concept[:120]}...")

            resume_markdown = f"""================================================================================
# {u_name.upper()}
Target Alignment: {target_profile.upper()}
Email: {u_email} | Phone: {u_phone} | Link: {u_links}
================================================================================

## 🎓 EDUCATION
- **{u_edu}**
  {u_univ} | {u_edu_year}
  Performance Matrix: {u_edu_metrics}

## 🛠️ TECHNICAL SKILLS
- **Core Domain Expertise:** {", ".join(selected_skills_list) if selected_skills_list else "Not Specified"}
- **Developer Tools & Frameworks:** {u_tools_frameworks if u_tools_frameworks else "Not Specified"}
"""

            if exp_enabled and u_job_title:
                resume_markdown += f"""
## 💼 WORK EXPERIENCE
- **{u_job_title}** | {u_company} ({u_job_duration})
  {u_job_desc}
"""

            resume_markdown += f"""
## 🚀 TECHNICAL PROJECTS
- **{u_proj_title}** [{u_proj_tech}]
  {u_proj_desc}
"""

            if dataset_project_bullets:
                resume_markdown += "\n- **Automated Matrix Compilations (Verified Local Data Items):**\n" + "\n".join(dataset_project_bullets) + "\n"

            resume_markdown += "\n================================================================================"

        st.balloons()
        st.markdown("### 📄 Generated Professional Blueprint")
        st.text(resume_markdown)
        
        st.download_button(
            label="📥 Download Structured Executive Resume (.txt)", 
            data=resume_markdown, 
            file_name=f"{u_name.replace(' ', '_')}_Executive_Resume.txt", 
            use_container_width=True
        )

# ==============================================================================
# SCREEN 4: CONTINUOUS LIVE INTERVIEW DECK LOOP
# ==============================================================================
elif st.session_state.app_mode in ["interview_topic", "interview_resume"]:
    q_idx = st.session_state.current_question_index
    qs = st.session_state.questions
    total_q = len(qs)
    
    if q_idx < total_q:
        st.subheader(f"Mock Evaluation Question {q_idx + 1} of {total_q}")
        current_q = qs[q_idx]
        st.info(f"❓ **{current_q['question']}** `(Target Tag: {current_q['skill'].upper()})`")
        
        TIME_LIMIT = 120 
        if st.session_state.timer_enabled:
            if st.session_state.start_time is None:
                st.session_state.start_time = time.time()
            elapsed_time = time.time() - st.session_state.start_time
            remaining_time = int(TIME_LIMIT - elapsed_time)
            
            if remaining_time <= 0:
                st.error("⚠️ Runtime Constraint: Time's Up!")
                st.session_state.answers[q_idx] = "Timeout!"
                
                clean_q = re.sub(r'(?i)\bvariant\s*\d+\b', '', current_q['question']).strip().capitalize()
                with st.spinner("Saving response token context..."):
                    st.session_state.ai_feedback_cache[q_idx] = get_ai_coaching_feedback(clean_q, current_q.get('answer', ''))
                    
                st.session_state.current_question_index += 1
                st.session_state.start_time = time.time()
                time.sleep(1.5)
                st.rerun()
            else:
                if remaining_time <= 15:
                    st.markdown(f"### 🛑 **Time Remaining: {remaining_time} seconds left!**")
                else:
                    st.markdown(f"### ⏱️ **Time Remaining: {remaining_time} seconds**")
                time.sleep(1)
                st.rerun()
        else:
            st.write("ℹ️ *Timer constraints disabled for this workspace session.*")
            
        user_ans = st.text_area("Type Your Answer Here:", key=f"ans_block_{q_idx}", placeholder="Draft your engineering response structure here...")
        
        if st.button("Submit Answer & Next ➡️", use_container_width=True):
            if user_ans.strip() == "" and not st.session_state.timer_enabled:
                st.warning("Please record textual insight tokens before proceeding.")
            else:
                st.session_state.answers[q_idx] = user_ans if user_ans.strip() != "" else "No answer provided."
                
                clean_q = re.sub(r'(?i)\bvariant\s*\d+\b', '', current_q['question']).strip().capitalize()
                with st.spinner("Processing AI evaluation data token..."):
                    st.session_state.ai_feedback_cache[q_idx] = get_ai_coaching_feedback(clean_q, current_q.get('answer', ''))
                
                st.session_state.current_question_index += 1
                st.session_state.start_time = time.time()
                st.rerun()
    else:
        st.balloons()
        st.markdown("<h2 style='color: #4ade80;'>🎉 Interview Completed!</h2>", unsafe_allow_html=True)
        st.subheader("Core AI Comprehensive Feedback & Answer Key Dashboard:")
        st.markdown("<div class='neon-hr'></div>", unsafe_allow_html=True)
        
        for i, q_obj in enumerate(qs):
            user_drafted = st.session_state.answers.get(i, "No entry logged.")
            clean_question = re.sub(r'(?i)\bvariant\s*\d+\b', '', q_obj['question']).strip()
            clean_question = re.sub(r'^[-\s\.\?:]+', '', clean_question).strip().capitalize()
            correct_ideal_answer = q_obj.get('answer', 'Baseline template error.')

            st.markdown(f"#### 📝 **Q{i+1}: {clean_question}**")
            st.markdown(f"<p style='color: #94a3b8;'>✏️ <b>Your Typed Response:</b> <i>\"{user_drafted}\"</i></p>", unsafe_allow_html=True)
            
            with st.expander("🔑 View Dataset Local Hint Reference", expanded=False):
                st.success(correct_ideal_answer)
            
            crisp_response = st.session_state.ai_feedback_cache.get(i, f"• {correct_ideal_answer}")
            
            st.markdown("##### 🎯 Ideal Professional Response (Interview-Ready):")
            st.info(crisp_response)
            st.markdown("<div class='neon-hr'></div>", unsafe_allow_html=True)
            
        if st.button("Reset & Return to Central Hub", use_container_width=True, type="primary"):
            st.session_state.app_mode = "home"
            st.session_state.questions = []
            st.session_state.current_question_index = 0
            st.session_state.answers = {}
            st.session_state.ai_feedback_cache = {}
            st.session_state.start_time = None
            st.rerun()