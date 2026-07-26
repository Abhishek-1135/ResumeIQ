import streamlit as st
import fitz 
import re
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


st.set_page_config(
    page_title="ResumeIQ",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1 {
        font-size: 42px !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 30px !important;
        font-weight: 650 !important;
    }

    h3 {
        font-size: 22px !important;
        font-weight: 600 !important;
    }

    .resume-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid rgba(128, 128, 128, 0.25);
    }

    </style>
    """,
    unsafe_allow_html=True
)


SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "node.js",
    "express",
    "django",
    "flask",
    "fastapi",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",
    "sqlite",
    "git",
    "github",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "nlp",
    "computer vision",
    "power bi",
    "tableau",
    "excel",
    "statistics",
    "rest api",
    "api",
    "oop",
    "object oriented programming",
    "linux",
    "cloud computing",
    "spark",
    "hadoop",
    "matplotlib",
    "seaborn",
    "streamlit",
    "communication",
    "problem solving",
    "teamwork",
    "leadership"
]


SKILL_ALIASES = {
    "python": ["python", "python3"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "react": ["react", "reactjs", "react.js"],
    "node.js": ["node.js", "nodejs", "node"],
    "sql": ["sql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo db"],
    "git": ["git"],
    "github": ["github"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud"],
    "machine learning": [
        "machine learning",
        "machine-learning"
    ],
    "deep learning": [
        "deep learning",
        "deep-learning"
    ],
    "artificial intelligence": [
        "artificial intelligence",
        "artificial intelligence",
        "ai"
    ],
    "data science": [
        "data science",
        "data-science"
    ],
    "data analysis": [
        "data analysis",
        "data-analysis"
    ],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": [
        "scikit-learn",
        "sklearn"
    ],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "keras": ["keras"],
    "nlp": [
        "nlp",
        "natural language processing"
    ],
    "computer vision": [
        "computer vision",
        "computer-vision"
    ],
    "power bi": [
        "power bi",
        "powerbi"
    ],
    "tableau": ["tableau"],
    "excel": [
        "excel",
        "microsoft excel"
    ],
    "rest api": [
        "rest api",
        "restful api",
        "rest"
    ],
    "oop": [
        "oop",
        "object oriented programming",
        "object-oriented programming"
    ],
    "streamlit": ["streamlit"]
}


ACTION_VERBS = [
    "developed",
    "designed",
    "built",
    "implemented",
    "created",
    "automated",
    "optimized",
    "analyzed",
    "managed",
    "deployed",
    "engineered",
    "integrated",
    "tested",
    "improved",
    "configured",
    "developed",
    "led"
]


STANDARD_SECTIONS = {
    "Profile / Summary": [
        "summary",
        "profile",
        "objective",
        "professional summary"
    ],
    "Skills": [
        "skills",
        "technical skills",
        "core skills"
    ],
    "Education": [
        "education",
        "academic"
    ],
    "Experience": [
        "experience",
        "work experience",
        "employment"
    ],
    "Projects": [
        "projects",
        "academic projects",
        "personal projects"
    ],
    "Certifications": [
        "certifications",
        "certificates"
    ]
}


def extract_text_from_pdf(uploaded_file):

    pdf_bytes = uploaded_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in document:

        text += page.get_text()

        text += "\n"

    document.close()

    return text.strip()


def extract_name(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines[:10]:

        if (
            "@" not in line
            and not re.search(
                r"\d",
                line
            )
            and len(line.split()) <= 5
            and len(line) > 2
        ):

            return line

    return "Not Detected"


def extract_email(text):

    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    if match:

        return match.group(0)

    return "Not Detected"


def extract_phone(text):

    match = re.search(
        r"(\+?\d[\d\s\-()]{8,}\d)",
        text
    )

    if match:

        return match.group(0).strip()

    return "Not Detected"


def detect_skills(text):

    text_lower = text.lower()

    found_skills = []

    for skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            pattern = r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)"

            if re.search(
                pattern,
                text_lower
            ):

                found_skills.append(
                    skill
                )

                break

    return list(
        dict.fromkeys(
            found_skills
        )
    )


def normalize_skills(skills):

    normalized = []

    for skill in skills:

        skill_lower = skill.lower().strip()

        if skill_lower in SKILL_ALIASES:

            normalized.append(
                skill_lower
            )

            continue

        for canonical, aliases in SKILL_ALIASES.items():

            if skill_lower in [
                alias.lower()
                for alias in aliases
            ]:

                normalized.append(
                    canonical
                )

                break

    return list(
        dict.fromkeys(
            normalized
        )
    )


def calculate_semantic_similarity(
    resume_text,
    job_text
):

    if not resume_text.strip() or not job_text.strip():

        return 0.0

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform(
            [
                resume_text,
                job_text
            ]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(
            similarity * 100,
            2
        )

    except Exception:

        return 0.0


def extract_keywords(text):

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b",
        text.lower()
    )

    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "you",
        "your",
        "will",
        "have",
        "has",
        "our",
        "their",
        "into",
        "about",
        "using",
        "use",
        "job",
        "work",
        "role",
        "candidate"
    }

    keywords = [
        word
        for word in words
        if word not in stopwords
    ]

    return list(
        dict.fromkeys(
            keywords
        )
    )


def extract_important_keywords(
    job_text,
    resume_text
):

    job_keywords = extract_keywords(
        job_text
    )

    resume_keywords = set(
        extract_keywords(
            resume_text
        )
    )

    important_keywords = []

    for keyword in job_keywords:

        if (
            keyword in resume_keywords
            or keyword in [
                skill.lower()
                for skill in SKILLS
            ]
        ):

            important_keywords.append(
                keyword
            )

    matched_keywords = [
        keyword
        for keyword in important_keywords
        if keyword in resume_keywords
    ]

    missing_keywords = [
        keyword
        for keyword in important_keywords
        if keyword not in resume_keywords
    ]

    return (
        important_keywords,
        matched_keywords,
        missing_keywords
    )


def analyze_resume_quality(
    text
):

    text_lower = text.lower()

    sections = {}

    for section_name, keywords in STANDARD_SECTIONS.items():

        sections[
            section_name
        ] = any(
            keyword in text_lower
            for keyword in keywords
        )

    section_score = (
        sum(
            sections.values()
        )
        /
        len(
            sections
        )
    ) * 100

    word_count = len(
        text.split()
    )

    if word_count >= 300:

        length_score = 100

    elif word_count >= 150:

        length_score = 80

    elif word_count >= 75:

        length_score = 60

    else:

        length_score = 30

    email_score = (
        100
        if extract_email(text) != "Not Detected"
        else 0
    )

    phone_score = (
        100
        if extract_phone(text) != "Not Detected"
        else 0
    )

    contact_score = (
        email_score
        +
        phone_score
    ) / 2

    quality_score = (
        section_score * 0.5
        +
        length_score * 0.2
        +
        contact_score * 0.3
    )

    return (
        sections,
        round(
            quality_score,
            2
        )
    )


def advanced_ats_analysis(
    resume_text,
    job_text
):

    resume_lower = resume_text.lower()

    job_lower = job_text.lower()

    job_skills = normalize_skills(
        detect_skills(
            job_text
        )
    )

    resume_skills = normalize_skills(
        detect_skills(
            resume_text
        )
    )

    matched_skills = [
        skill
        for skill in job_skills
        if skill in resume_skills
    ]

    missing_skills = [
        skill
        for skill in job_skills
        if skill not in resume_skills
    ]

    if job_skills:

        skill_match = (
            len(
                matched_skills
            )
            /
            len(
                job_skills
            )
        ) * 100

    else:

        skill_match = 0

    (
        important_keywords,
        matched_keywords,
        missing_keywords
    ) = extract_important_keywords(
        job_text,
        resume_text
    )

    if important_keywords:

        keyword_match = (
            len(
                matched_keywords
            )
            /
            len(
                important_keywords
            )
        ) * 100

    else:

        keyword_match = 0

    detected_action_verbs = []

    for verb in ACTION_VERBS:

        if re.search(
            r"(?<!\w)"
            + re.escape(verb)
            + r"(?!\w)",
            resume_lower
        ):

            detected_action_verbs.append(
                verb
            )

    quantified_count = len(
        re.findall(
            r"\b\d+(?:\.\d+)?%?\b",
            resume_text
        )
    )

    (
        sections,
        quality_score
    ) = analyze_resume_quality(
        resume_text
    )

    semantic_score = calculate_semantic_similarity(
        resume_text,
        job_text
    )

    ats_score = (
        skill_match * 0.35
        +
        keyword_match * 0.25
        +
        semantic_score * 0.20
        +
        quality_score * 0.20
    )

    return {

        "ats_score": round(
            ats_score,
            2
        ),

        "skill_match": round(
            skill_match,
            2
        ),

        "keyword_match": round(
            keyword_match,
            2
        ),

        "semantic_score": round(
            semantic_score,
            2
        ),

        "quality_score": round(
            quality_score,
            2
        ),

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "important_keywords": important_keywords,

        "matched_keywords": matched_keywords,

        "missing_keywords": missing_keywords,

        "detected_action_verbs": detected_action_verbs,

        "quantified_count": quantified_count,

        "sections": sections

    }


def analyze_single_job(
    resume_text,
    job_text,
    job_name
):

    resume_skills = normalize_skills(
        detect_skills(
            resume_text
        )
    )

    job_skills = normalize_skills(
        detect_skills(
            job_text
        )
    )

    matched_skills = [
        skill
        for skill in job_skills
        if skill in resume_skills
    ]

    missing_skills = [
        skill
        for skill in job_skills
        if skill not in resume_skills
    ]

    if job_skills:

        skill_score = (
            len(
                matched_skills
            )
            /
            len(
                job_skills
            )
        ) * 100

    else:

        skill_score = 0

    semantic_score = calculate_semantic_similarity(
        resume_text,
        job_text
    )

    (
        important_keywords,
        matched_keywords,
        missing_keywords
    ) = extract_important_keywords(
        job_text,
        resume_text
    )

    if important_keywords:

        keyword_score = (
            len(
                matched_keywords
            )
            /
            len(
                important_keywords
            )
        ) * 100

    else:

        keyword_score = 0

    (
        sections,
        quality_score
    ) = analyze_resume_quality(
        resume_text
    )

    ats = advanced_ats_analysis(
        resume_text,
        job_text
    )

    overall_score = (
        skill_score * 0.35
        +
        semantic_score * 0.25
        +
        keyword_score * 0.20
        +
        quality_score * 0.20
    )

    return {

        "job_name": job_name,

        "overall_score": round(
            overall_score,
            2
        ),

        "skill_score": round(
            skill_score,
            2
        ),

        "semantic_score": round(
            semantic_score,
            2
        ),

        "keyword_score": round(
            keyword_score,
            2
        ),

        "ats_score": ats[
            "ats_score"
        ],

        "quality_score": quality_score,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "matched_keywords": matched_keywords,

        "missing_keywords": missing_keywords

    }


def generate_skill_recommendations(
    missing_skills
):

    recommendations = []

    for skill in missing_skills:

        recommendations.append(
            f"Consider learning or strengthening {skill} "
            f"through projects, courses, and practical exercises."
        )

    return recommendations


def generate_resume_improvement_suggestions(
    resume_text,
    job_text
):

    suggestions = []

    ats = advanced_ats_analysis(
        resume_text,
        job_text
    )

    if ats[
        "quality_score"
    ] < 70:

        suggestions.append(
            "Improve the overall structure and completeness of your resume."
        )

    if len(
        ats[
            "detected_action_verbs"
        ]
    ) < 5:

        suggestions.append(
            "Use stronger action verbs such as Developed, Built, "
            "Implemented, Designed, Optimized, and Deployed."
        )

    if ats[
        "quantified_count"
    ] < 3:

        suggestions.append(
            "Add measurable achievements using numbers, percentages, "
            "performance improvements, or project scale."
        )

    if ats[
        "missing_keywords"
    ]:

        suggestions.append(
            "Naturally include relevant job-description keywords "
            "where they accurately describe your experience."
        )

    if ats[
        "missing_skills"
    ]:

        suggestions.append(
            "Consider developing the missing technical skills "
            "identified in the job description."
        )

    return suggestions


def recommend_job_roles(
    resume_text
):

    skills = normalize_skills(
        detect_skills(
            resume_text
        )
    )

    roles = []

    if any(
        skill in skills
        for skill in [
            "python",
            "django",
            "flask",
            "fastapi"
        ]
    ):

        roles.append(
            "Python Developer"
        )

    if any(
        skill in skills
        for skill in [
            "machine learning",
            "scikit-learn",
            "tensorflow",
            "pytorch"
        ]
    ):

        roles.append(
            "Machine Learning Engineer"
        )

    if any(
        skill in skills
        for skill in [
            "data science",
            "pandas",
            "numpy",
            "statistics"
        ]
    ):

        roles.append(
            "Data Scientist"
        )

    if any(
        skill in skills
        for skill in [
            "data analysis",
            "power bi",
            "tableau",
            "excel"
        ]
    ):

        roles.append(
            "Data Analyst"
        )

    if any(
        skill in skills
        for skill in [
            "javascript",
            "react",
            "html",
            "css"
        ]
    ):

        roles.append(
            "Frontend Developer"
        )

    if any(
        skill in skills
        for skill in [
            "node.js",
            "express",
            "sql",
            "mongodb"
        ]
    ):

        roles.append(
            "Backend Developer"
        )

    if any(
        skill in skills
        for skill in [
            "java",
            "spring"
        ]
    ):

        roles.append(
            "Java Developer"
        )

    if not roles:

        roles.append(
            "Software Developer"
        )

    return list(
        dict.fromkeys(
            roles
        )
    )


def generate_pdf_report(
    candidate_name,
    candidate_email,
    candidate_phone,
    best_job,
    comparison_results,
    skill_recommendations,
    improvement_suggestions,
    recommended_roles
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles[
        "Title"
    ]

    title_style.alignment = TA_CENTER

    story = []

    story.append(
        Paragraph(
            "ResumeIQ Analysis Report",
            title_style
        )
    )

    story.append(
        Spacer(
            1,
            20
        )
    )

    story.append(
        Paragraph(
            f"<b>Candidate Name:</b> {candidate_name}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Email:</b> {candidate_email}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Phone:</b> {candidate_phone}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "<b>Best Job Recommendation</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"Job: {best_job['job_name']}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Overall Match Score: {best_job['overall_score']}%",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Skill Match: {best_job['skill_score']}%",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Semantic Match: {best_job['semantic_score']}%",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Keyword Match: {best_job['keyword_score']}%",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"ATS Score: {best_job['ats_score']}%",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "<b>Job Comparison</b>",
            styles["Heading2"]
        )
    )

    table_data = [
        [
            "Job",
            "Overall",
            "Skill",
            "Semantic",
            "Keyword",
            "ATS"
        ]
    ]

    for result in comparison_results:

        table_data.append(
            [
                result[
                    "job_name"
                ],
                f"{result['overall_score']}%",
                f"{result['skill_score']}%",
                f"{result['semantic_score']}%",
                f"{result['keyword_score']}%",
                f"{result['ats_score']}%"
            ]
        )

    table = Table(
        table_data,
        repeatRows=1
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.grey
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                )
            ]
        )
    )

    story.append(
        table
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "<b>Missing Skills</b>",
            styles["Heading2"]
        )
    )

    if best_job[
        "missing_skills"
    ]:

        for skill in best_job[
            "missing_skills"
        ]:

            story.append(
                Paragraph(
                    f"• {skill}",
                    styles["Normal"]
                )
            )

    else:

        story.append(
            Paragraph(
                "No major missing skills detected.",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "<b>Skill Improvement Recommendations</b>",
            styles["Heading2"]
        )
    )

    for recommendation in skill_recommendations:

        story.append(
            Paragraph(
                f"• {recommendation}",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "<b>Resume Improvement Suggestions</b>",
            styles["Heading2"]
        )
    )

    for suggestion in improvement_suggestions:

        story.append(
            Paragraph(
                f"• {suggestion}",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "<b>Recommended Career Roles</b>",
            styles["Heading2"]
        )
    )

    for role in recommended_roles:

        story.append(
            Paragraph(
                f"• {role}",
                styles["Normal"]
            )
        )

    document.build(
        story
    )

    buffer.seek(
        0
    )

    return buffer


st.markdown(
    """
    <div style="text-align: center;">

    <h1>📄 ResumeIQ</h1>

    <p style="font-size: 20px;">
    AI-Powered Resume Analysis & Job Matching Platform
    </p>

    <p>
    Analyze your resume • Match jobs • Identify skill gaps • Improve your career opportunities
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


st.sidebar.title(
    "📄 ResumeIQ"
)

st.sidebar.write(
    "Your Intelligent Resume Assistant"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "📄 Resume Analysis",
        "💼 Job Matching",
        "📊 Skill Gap Analysis",
        "📈 ATS Analysis",
        "📑 Reports",
        "ℹ️ About ResumeIQ"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "ResumeIQ analyzes your resume, compares it with job descriptions, "
    "identifies skill gaps, calculates ATS compatibility, and recommends "
    "the best matching opportunities."
)


if "analysis_done" not in st.session_state:

    st.session_state.analysis_done = False


if "comparison_results" not in st.session_state:

    st.session_state.comparison_results = []


if "best_job" not in st.session_state:

    st.session_state.best_job = None


if "resume_text" not in st.session_state:

    st.session_state.resume_text = ""


if "job_data" not in st.session_state:

    st.session_state.job_data = []


if "candidate_name" not in st.session_state:

    st.session_state.candidate_name = "Not Detected"


if "candidate_email" not in st.session_state:

    st.session_state.candidate_email = "Not Detected"


if "candidate_phone" not in st.session_state:

    st.session_state.candidate_phone = "Not Detected"


if "skill_recommendations" not in st.session_state:

    st.session_state.skill_recommendations = []


if "improvement_suggestions" not in st.session_state:

    st.session_state.improvement_suggestions = []


if "recommended_roles" not in st.session_state:

    st.session_state.recommended_roles = []


if page == "🏠 Dashboard":
    st.title("🚀 Welcome to ResumeIQ")

    st.subheader("AI-Powered Resume Analysis & Job Matching Platform")

    st.write(
        "ResumeIQ helps you analyze your resume, evaluate job compatibility, "
        "identify skill gaps, check ATS compatibility, and generate detailed "
        "career recommendations."
    )

    st.divider()

    st.header("🎯 What Can ResumeIQ Do?")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume Analysis")
        st.write(
            "Analyze your resume and identify important sections, skills, "
            "education, experience, projects, and certifications."
        )

        st.subheader("📊 Skill Gap Analysis")
        st.write(
            "Discover missing skills and get personalized recommendations "
            "to improve your career profile."
        )

    with col2:
        st.subheader("💼 Job Matching")
        st.write(
            "Compare your resume with a job description and understand "
            "how well your skills match the role."
        )

        st.subheader("📈 ATS Analysis")
        st.write(
            "Evaluate your resume for ATS compatibility and identify areas "
            "that can be improved."
        )

    st.divider()

    st.header("🔄 How ResumeIQ Works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.subheader("1️⃣ Upload")
        st.write("Upload your resume in PDF format.")

    with step2:
        st.subheader("2️⃣ Analyze")
        st.write("ResumeIQ extracts and analyzes your resume content.")

    with step3:
        st.subheader("3️⃣ Compare")
        st.write("Compare your resume with a target job description.")

    with step4:
        st.subheader("4️⃣ Improve")
        st.write("Get skill gap insights and personalized recommendations.")

    st.divider()

    st.header("🛠️ Technologies Used")

    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

    with tech_col1:
        st.write("🐍 Python")
        st.write("⚡ Streamlit")

    with tech_col2:
        st.write("📄 PyMuPDF")
        st.write("📊 Pandas")

    with tech_col3:
        st.write("🤖 Scikit-learn")
        st.write("🔢 NumPy")

    with tech_col4:
        st.write("📈 Plotly")
        st.write("📑 ReportLab")

    st.divider()

    st.header("🚀 Start Using ResumeIQ")

    st.write(
        "Use the sidebar to navigate through Resume Analysis, Job Matching, "
        "Skill Gap Analysis, ATS Analysis, and Reports."
    )

    st.info(
        "💡 Tip: Start by uploading your resume in the Resume Analysis section."
    )

    st.divider()

    st.caption(
        "ResumeIQ v1.0 | AI-Powered Resume Analysis & Job Matching Platform"
    )


if page == "📄 Resume Analysis":

    st.header(
        "📄 Resume Analysis"
    )

    st.markdown(
        """
        <div class="resume-card">

        <h3>📄 Upload Your Resume</h3>

        <p>
        Upload your resume in PDF format to begin your analysis.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    resume_file = st.file_uploader(
        "Choose your resume",
        type=["pdf"],
        key="main_resume"
    )

    if resume_file:

        if st.button(
            "🔍 Analyze Resume",
            type="primary"
        ):

            progress = st.progress(
                0
            )

            status = st.empty()

            status.write(
                "📄 Extracting resume text..."
            )

            resume_text = extract_text_from_pdf(
                resume_file
            )

            progress.progress(
                30
            )

            if not resume_text.strip():

                st.error(
                    "Could not extract text from this PDF."
                )

            else:

                status.write(
                    "🔍 Detecting candidate information..."
                )

                candidate_name = extract_name(
                    resume_text
                )

                candidate_email = extract_email(
                    resume_text
                )

                candidate_phone = extract_phone(
                    resume_text
                )

                progress.progress(
                    60
                )

                status.write(
                    "🧠 Analyzing resume quality and skills..."
                )

                skills = normalize_skills(
                    detect_skills(
                        resume_text
                    )
                )

                sections, quality_score = analyze_resume_quality(
                    resume_text
                )

                progress.progress(
                    100
                )

                status.success(
                    "✅ Resume analysis complete!"
                )

                st.session_state.resume_text = resume_text

                st.session_state.candidate_name = candidate_name

                st.session_state.candidate_email = candidate_email

                st.session_state.candidate_phone = candidate_phone

                st.session_state.resume_skills = skills

                st.session_state.resume_sections = sections

                st.session_state.resume_quality = quality_score

        if st.session_state.resume_text:

            st.divider()

            st.subheader(
                "👤 Candidate Information"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Name",
                    st.session_state.candidate_name
                )

            with col2:

                st.metric(
                    "Email",
                    st.session_state.candidate_email
                )

            with col3:

                st.metric(
                    "Phone",
                    st.session_state.candidate_phone
                )

            st.divider()

            st.subheader(
                "🛠️ Detected Skills"
            )

            skills = st.session_state.get(
                "resume_skills",
                []
            )

            if skills:

                st.write(
                    ", ".join(
                        skills
                    )
                )

            else:

                st.warning(
                    "No skills detected."
                )

            st.subheader(
                "📋 Resume Sections"
            )

            sections = st.session_state.get(
                "resume_sections",
                {}
            )

            for section, present in sections.items():

                if present:

                    st.success(
                        f"✅ {section}"
                    )

                else:

                    st.warning(
                        f"⚠️ {section} not detected"
                    )

            st.metric(
                "📊 Resume Quality Score",
                f"{st.session_state.get('resume_quality', 0):.2f}%"
            )


if page == "💼 Job Matching":

    st.header(
        "💼 Resume vs Job Matching"
    )

    if not st.session_state.resume_text:

        st.warning(
            "Please analyze your resume first from the Resume Analysis section."
        )

    else:

        st.markdown(
            """
            <div class="resume-card">

            <h3>💼 Upload Job Descriptions</h3>

            <p>
            Upload one or more job descriptions to compare your resume.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        job_files = st.file_uploader(
            "Choose job descriptions",
            type=["pdf"],
            accept_multiple_files=True,
            key="job_descriptions"
        )

        if job_files:

            if st.button(
                "🚀 Analyze All Jobs",
                type="primary"
            ):

                progress = st.progress(
                    0
                )

                status = st.empty()

                job_data = []

                total_jobs = len(
                    job_files
                )

                for index, job_file in enumerate(
                    job_files
                ):

                    status.write(
                        f"🔍 Analyzing job {index + 1} of {total_jobs}..."
                    )

                    job_text = extract_text_from_pdf(
                        job_file
                    )

                    job_name = job_file.name.rsplit(
                        ".",
                        1
                    )[0]

                    if job_text.strip():

                        job_data.append(
                            {
                                "name": job_name,
                                "text": job_text
                            }
                        )

                    progress.progress(
                        int(
                            (
                                index + 1
                            )
                            /
                            total_jobs
                            *
                            100
                        )
                    )

                comparison_results = []

                for job in job_data:

                    result = analyze_single_job(
                        st.session_state.resume_text,
                        job["text"],
                        job["name"]
                    )

                    comparison_results.append(
                        result
                    )

                comparison_results.sort(
                    key=lambda x: x[
                        "overall_score"
                    ],
                    reverse=True
                )

                st.session_state.job_data = job_data

                st.session_state.comparison_results = comparison_results

                if comparison_results:

                    st.session_state.best_job = comparison_results[
                        0
                    ]

                    best_job = comparison_results[
                        0
                    ]

                    best_job_text = ""

                    for job in job_data:

                        if job[
                            "name"
                        ] == best_job[
                            "job_name"
                        ]:

                            best_job_text = job[
                                "text"
                            ]

                            break

                    st.session_state.best_job_text = best_job_text

                    st.session_state.skill_recommendations = generate_skill_recommendations(
                        best_job[
                            "missing_skills"
                        ]
                    )

                    st.session_state.improvement_suggestions = generate_resume_improvement_suggestions(
                        st.session_state.resume_text,
                        best_job_text
                    )

                    st.session_state.recommended_roles = recommend_job_roles(
                        st.session_state.resume_text
                    )

                    st.session_state.analysis_done = True

                    status.success(
                        "✅ Job analysis complete!"
                    )

            if st.session_state.comparison_results:

                best_job = st.session_state.best_job

                st.divider()

                st.subheader(
                    "📊 Resume Performance"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "🎯 Overall Match",
                        f"{best_job['overall_score']:.2f}%"
                    )

                with col2:

                    st.metric(
                        "🛠️ Skill Match",
                        f"{best_job['skill_score']:.2f}%"
                    )

                with col3:

                    st.metric(
                        "🔎 Keyword Match",
                        f"{best_job['keyword_score']:.2f}%"
                    )

                with col4:

                    st.metric(
                        "📄 ATS Score",
                        f"{best_job['ats_score']:.2f}%"
                    )

                if best_job[
                    "overall_score"
                ] >= 80:

                    st.success(
                        "🏆 Excellent match! Your resume is highly aligned with this job."
                    )

                elif best_job[
                    "overall_score"
                ] >= 65:

                    st.info(
                        "👍 Good match! A few improvements could make your resume stronger."
                    )

                elif best_job[
                    "overall_score"
                ] >= 50:

                    st.warning(
                        "⚠️ Moderate match. Consider improving your skills and keywords."
                    )

                else:

                    st.error(
                        "❌ Low match. Your resume needs significant tailoring for this role."
                    )

                st.divider()

                st.subheader(
                    "🏆 Best Job Recommendation"
                )

                st.success(
                    f"🏆 {best_job['job_name']} — "
                    f"{best_job['overall_score']:.2f}% Match"
                )

                with st.expander(
                    "❌ View Missing Skills"
                ):

                    if best_job[
                        "missing_skills"
                    ]:

                        for skill in best_job[
                            "missing_skills"
                        ]:

                            st.write(
                                f"🔴 {skill}"
                            )

                    else:

                        st.success(
                            "No major missing skills detected."
                        )

                with st.expander(
                    "✅ View Matched Skills"
                ):

                    if best_job[
                        "matched_skills"
                    ]:

                        for skill in best_job[
                            "matched_skills"
                        ]:

                            st.write(
                                f"🟢 {skill}"
                            )

                    else:

                        st.info(
                            "No matched skills detected."
                        )

                st.divider()

                st.subheader(
                    "📈 Job Match Ranking"
                )

                ranking_data = []

                for result in st.session_state.comparison_results:

                    ranking_data.append(
                        {
                            "Job": result[
                                "job_name"
                            ],
                            "Overall Match": f"{result['overall_score']:.2f}%",
                            "Skill Match": f"{result['skill_score']:.2f}%",
                            "Semantic Match": f"{result['semantic_score']:.2f}%",
                            "Keyword Match": f"{result['keyword_score']:.2f}%",
                            "ATS Score": f"{result['ats_score']:.2f}%"
                        }
                    )

                st.dataframe(
                    ranking_data,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader(
                    "📊 Score Comparison"
                )

                chart_data = {

                    "Job": [
                        result[
                            "job_name"
                        ]
                        for result in st.session_state.comparison_results
                    ],

                    "Overall Match": [
                        result[
                            "overall_score"
                        ]
                        for result in st.session_state.comparison_results
                    ],

                    "ATS Score": [
                        result[
                            "ats_score"
                        ]
                        for result in st.session_state.comparison_results
                    ]

                }

                st.bar_chart(
                    chart_data,
                    x="Job",
                    y=[
                        "Overall Match",
                        "ATS Score"
                    ]
                )


if page == "📊 Skill Gap Analysis":

    st.header(
        "🎯 Skill Gap Analysis"
    )

    if not st.session_state.analysis_done:

        st.warning(
            "Please complete the job matching analysis first."
        )

    else:

        best_job = st.session_state.best_job

        st.subheader(
            "❌ Skills to Improve"
        )

        missing_skills = best_job[
            "missing_skills"
        ]

        if missing_skills:

            for skill in missing_skills:

                st.warning(
                    f"⚠️ {skill}"
                )

        else:

            st.success(
                "🎉 No major skill gaps detected!"
            )

        st.divider()

        st.subheader(
            "💡 Skill Improvement Recommendations"
        )

        for recommendation in st.session_state.skill_recommendations:

            st.info(
                recommendation
            )

        st.divider()

        st.subheader(
            "👨‍💻 Recommended Career Roles"
        )

        for role in st.session_state.recommended_roles:

            st.success(
                f"🎯 {role}"
            )


if page == "📈 ATS Analysis":

    st.header(
        "📈 ATS Analysis"
    )

    if not st.session_state.analysis_done:

        st.warning(
            "Please complete the job matching analysis first."
        )

    else:

        best_job = st.session_state.best_job

        job_text = st.session_state.best_job_text

        ats = advanced_ats_analysis(
            st.session_state.resume_text,
            job_text
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "ATS Score",
                f"{ats['ats_score']:.2f}%"
            )

        with col2:

            st.metric(
                "Semantic Score",
                f"{ats['semantic_score']:.2f}%"
            )

        with col3:

            st.metric(
                "Keyword Match",
                f"{ats['keyword_match']:.2f}%"
            )

        with col4:

            st.metric(
                "Resume Quality",
                f"{ats['quality_score']:.2f}%"
            )

        st.divider()

        st.subheader(
            "🔎 Keyword Analysis"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### ✅ Matched Keywords"
            )

            for keyword in ats[
                "matched_keywords"
            ]:

                st.success(
                    keyword
                )

        with col2:

            st.markdown(
                "### ❌ Missing Keywords"
            )

            for keyword in ats[
                "missing_keywords"
            ]:

                st.warning(
                    keyword
                )

        st.divider()

        st.subheader(
            "💪 Action Verbs"
        )

        if ats[
            "detected_action_verbs"
        ]:

            st.write(
                ", ".join(
                    ats[
                        "detected_action_verbs"
                    ]
                )
            )

        else:

            st.warning(
                "No strong action verbs detected."
            )

        st.subheader(
            "📊 Quantified Achievements"
        )

        st.metric(
            "Detected Quantified Items",
            ats[
                "quantified_count"
            ]
        )

        st.divider()

        st.subheader(
            "💡 Resume Improvement Suggestions"
        )

        for suggestion in st.session_state.improvement_suggestions:

            st.info(
                suggestion
            )


if page == "📥 Reports":

    st.header(
        "📥 ResumeIQ Reports"
    )

    if not st.session_state.analysis_done:

        st.warning(
            "Please complete the resume and job analysis first."
        )

    else:

        best_job = st.session_state.best_job

        report = generate_pdf_report(

            st.session_state.candidate_name,

            st.session_state.candidate_email,

            st.session_state.candidate_phone,

            best_job,

            st.session_state.comparison_results,

            st.session_state.skill_recommendations,

            st.session_state.improvement_suggestions,

            st.session_state.recommended_roles

        )

        st.success(
            "✅ Your ResumeIQ report is ready."
        )

        st.download_button(

            label="📥 Download ResumeIQ Analysis Report",

            data=report,

            file_name="ResumeIQ_Analysis_Report.pdf",

            mime="application/pdf",

            type="primary"

        )

        st.divider()

        st.subheader(
            "📋 Report Summary"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Best Job",
                best_job[
                    "job_name"
                ]
            )

        with col2:

            st.metric(
                "Overall Score",
                f"{best_job['overall_score']:.2f}%"
            )

        with col3:

            st.metric(
                "ATS Score",
                f"{best_job['ats_score']:.2f}%"
            )
elif page == "ℹ️ About ResumeIQ":
    st.title("ℹ️ About ResumeIQ")

    st.markdown("""
    ## ResumeIQ

    **AI-Powered Resume Analysis & Job Matching Platform**

    ResumeIQ is an intelligent career assistance platform designed to help
    job seekers analyze their resumes, understand their strengths, identify
    missing skills, evaluate ATS compatibility, and discover relevant job
    opportunities.

    ### 🚀 Key Features

    - 📄 Resume PDF Analysis
    - 💼 Job Matching
    - 📊 Skill Gap Analysis
    - 📈 ATS Compatibility Analysis
    - 📑 Professional PDF Reports
    - 🎯 Resume Quality Scoring
    - 🔍 Matched and Missing Skill Detection
    - 💡 Personalized Skill Improvement Recommendations

    ### 🛠️ Technologies Used

    - Python
    - Streamlit
    - PyMuPDF
    - Pandas
    - NumPy
    - Scikit-learn
    - Plotly
    - Python-docx
    - OpenPyXL
    - ReportLab

    ### 🔄 How ResumeIQ Works

    1. Upload your resume in PDF format.
    2. ResumeIQ extracts and analyzes the resume content.
    3. The system identifies important resume sections and skills.
    4. Job descriptions are compared with the candidate profile.
    5. Missing skills and skill gaps are identified.
    6. ATS compatibility is evaluated.
    7. A detailed report with recommendations is generated.

    ### 🎓 Project Information

    **Project:** ResumeIQ

    **Type:** B.Tech Computer Science Engineering Project

    **Purpose:** AI-powered resume analysis and job matching

    **Version:** 1.0

    ### 👨‍💻 Project Status

    ✅ Successfully deployed and publicly accessible

    ---
    
    **ResumeIQ v1.0**

    AI-Powered Resume Analysis & Job Matching Platform
    """)

st.markdown(
    """
    <div class="footer">

    <h3>📄 ResumeIQ</h3>

    <p>
    AI-Powered Resume Analysis & Job Matching Platform
    </p>

    <p>
    Built as a B.Tech Computer Science Engineering Project
    </p>

    <p>
    © 2026 ResumeIQ
    </p>

    </div>
    """,
    unsafe_allow_html=True
)