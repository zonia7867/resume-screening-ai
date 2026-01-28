"""
Resume Screening Automation 
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import io
import base64
import json
import re

# Import custom modules
from src.data_preprocessing import ResumePreprocessor
from src.feature_extraction import FeatureExtractor, ResumeMatcher

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'page_loaded' not in st.session_state:
    st.session_state.page_loaded = False
if 'shortlisted_candidates' not in st.session_state:
    st.session_state.shortlisted_candidates = []
if 'rejected_candidates' not in st.session_state:
    st.session_state.rejected_candidates = []
if 'selected_candidate' not in st.session_state:
    st.session_state.selected_candidate = None

# ============================================================================
#  CSS
# ============================================================================

st.markdown("""
    <style>
    /* Import Fonts and Icons */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* CSS Variables */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --primary-light: #818cf8;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1e293b;
        --gray: #64748b;
        --light-gray: #f1f5f9;
        --white: #ffffff;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    
    /* Hide default radio buttons */
    [data-testid="stSidebar"] .stRadio {
        display: none;
    }
    
    /* Modern Cards */
    .modern-card {
        background: var(--white);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: var(--transition);
        animation: scaleIn 0.3s ease-out;
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--white) 0%, #fafafa 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: var(--white);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Skill Badges */
    .skill-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: var(--white);
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: var(--transition);
    }
    
    .skill-badge:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Status Badges */
    .status-badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
    }
    
    .status-danger {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--light-gray);
        padding: 4px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--white);
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid var(--light-gray);
        transition: var(--transition);
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Headers */
    h1 {
        color: var(--dark);
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: var(--dark);
        font-weight: 600;
        font-size: 1.75rem;
    }
    
    h3 {
        color: var(--gray);
        font-weight: 600;
        font-size: 1.25rem;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--white);
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
        transition: var(--transition);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--light-gray);
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Filter Panel */
    .filter-panel {
        background: var(--white);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 0, 0, 0.05);
        animation: slideUp 0.3s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-gray);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 10px;
    }
    
    /* Navigation Buttons */
    .nav-button {
        width: 100%;
        padding: 12px 16px;
        margin: 4px 0;
        background: transparent;
        border: none;
        border-radius: 10px;
        color: #cbd5e1;
        font-size: 0.95rem;
        font-weight: 500;
        text-align: left;
        cursor: pointer;
        transition: var(--transition);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3);
    }
    
    .nav-button i {
        width: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(file_path):
    """Load resume data from CSV"""
    try:
        if isinstance(file_path, str):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        if os.path.exists('models/tfidf_vectorizer.pkl') and os.path.exists('models/label_encoder.pkl'):
            extractor = FeatureExtractor()
            extractor.load_vectorizer('models/tfidf_vectorizer.pkl')
            extractor.load_label_encoder('models/label_encoder.pkl')
            return extractor
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return None

def export_to_csv(df):
    """Export dataframe to CSV"""
    return df.to_csv(index=False).encode('utf-8')

def export_to_excel(df, filename="resumes_export.xlsx"):
    """Export dataframe to Excel - with fallback to CSV"""
    try:
        import openpyxl
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Resumes')
        output.seek(0)
        return output, "xlsx"
    except ImportError:
        st.warning("Note: Install openpyxl for Excel export (pip install openpyxl). Downloading as CSV instead.")
        return df.to_csv(index=False).encode('utf-8'), "csv"
    except Exception as e:
        st.error(f"Export error: {e}")
        return None, None

def extract_experience_years(text):
    """Extract years of experience from resume text"""
    if pd.isna(text):
        return None
    
    text = str(text).lower()
    patterns = [
        r'(\d+)\+?\s*(?:to\s*(\d+))?\s*years?\s*(?:of\s*)?experience',
        r'(\d+)\+?\s*(?:to\s*(\d+))?\s*yrs',
        r'experience\s*:?\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*in',
    ]
    
    max_years = 0
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    years = int(match[0]) if match[0] else 0
                    max_years = max(max_years, years)
                else:
                    years = int(match)
                    max_years = max(max_years, years)
    
    return max_years if max_years > 0 else None

def extract_education_level(text):
    """Extract education level from resume text"""
    if pd.isna(text):
        return "Unknown"
    
    text = str(text).lower()
    
    if any(keyword in text for keyword in ['phd', 'ph.d', 'doctorate', 'doctoral']):
        return "PhD"
    elif any(keyword in text for keyword in ['master', 'm.s', 'm.tech', 'mba', 'm.e', 'msc']):
        return "Master's"
    elif any(keyword in text for keyword in ['bachelor', 'b.s', 'b.tech', 'b.e', 'bsc', 'undergraduate']):
        return "Bachelor's"
    elif any(keyword in text for keyword in ['diploma', 'associate']):
        return "Diploma"
    else:
        return "Unknown"

# ============================================================================
# SIDEBAR 
# ============================================================================

def render_sidebar():
    """Render sidebar with button navigation"""
    
    with st.sidebar:
        # Header
        st.markdown("""
            <div style="text-align: center; padding: 2rem 0 1rem 0;">
                <i class="fas fa-bullseye" style="font-size: 3rem; color: #818cf8;"></i>
                <h2 style="color: #818cf8; margin: 1rem 0 0 0; font-weight: 700;">Resume AI</h2>
                <p style="color: #94a3b8; font-size: 0.875rem; margin: 0.5rem 0 0 0;">
                    Intelligent Recruitment Platform
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigation Buttons
        st.markdown("### <i class='fas fa-bars'></i> Navigation", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        pages = {
            "Home": "fa-home",
            "Resume Matching": "fa-users-rectangle",
            "Data Analysis": "fa-chart-line",
            "Category Insights": "fa-magnifying-glass-chart",
            "Settings": "fa-gear"
        }
        
        selected_page = st.session_state.current_page
        
        for page_name, icon in pages.items():
            active_class = "active" if selected_page == page_name else ""
            
            button_html = f"""
                <button class="nav-button {active_class}" onclick="return false;">
                    <i class="fas {icon}"></i>
                    <span>{page_name}</span>
                </button>
            """
            
            if st.button(page_name, key=f"nav_{page_name}", use_container_width=True):
                st.session_state.current_page = page_name
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # File Upload
        st.markdown("### <i class='fas fa-file-arrow-up'></i> Data Upload", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Resume Dataset", 
            type=['csv'], 
            key="file_uploader",
            help="Upload a CSV file containing resume data"
        )
        
        if uploaded_file:
            st.success("File loaded successfully")
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### <i class='fas fa-chart-simple'></i> Quick Stats", unsafe_allow_html=True)
        
        if 'stats' in st.session_state:
            stats = st.session_state.stats
            
            st.markdown(f"""
                <div class="modern-card" style="margin: 0.5rem 0;">
                    <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">
                        <i class="fas fa-file-lines"></i> Total Resumes
                    </div>
                    <div style="color: #1e293b; font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem;">
                        {stats.get('total', 0):,}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="modern-card" style="margin: 0.5rem 0;">
                        <div style="color: #10b981; font-size: 1.25rem; font-weight: 700;">
                            <i class="fas fa-star"></i> {len(st.session_state.shortlisted_candidates)}
                        </div>
                        <div style="color: #64748b; font-size: 0.7rem; margin-top: 0.25rem;">Shortlisted</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="modern-card" style="margin: 0.5rem 0;">
                        <div style="color: #ef4444; font-size: 1.25rem; font-weight: 700;">
                            <i class="fas fa-xmark"></i> {len(st.session_state.rejected_candidates)}
                        </div>
                        <div style="color: #64748b; font-size: 0.7rem; margin-top: 0.25rem;">Rejected</div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
            <div style="text-align: center; color: #64748b; font-size: 0.75rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <p style="margin: 0;"><i class="fas fa-robot"></i> Powered by AI</p>
                <p style="margin: 0.25rem 0 0 0;">v2.6 Professional</p>
            </div>
        """, unsafe_allow_html=True)
        
        return st.session_state.current_page, uploaded_file

# ============================================================================
# HOME PAGE
# ============================================================================

def render_home_page(df, df_processed):
    """Render modern home page"""
    
    st.markdown("<h1><i class='fas fa-house'></i> Dashboard</h1>", unsafe_allow_html=True)
    
    if df is None:
        st.markdown("""
            <div class="modern-card" style="text-align: center; padding: 4rem 2rem;">
                <i class="fas fa-hand-wave" style="font-size: 4rem; color: #6366f1; margin-bottom: 1rem;"></i>
                <h2 style="color: #1e293b; margin-bottom: 1rem;">Welcome to Resume AI</h2>
                <p style="color: #64748b; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                    Upload your resume dataset to get started with intelligent candidate matching
                    and advanced analytics.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Store stats
    st.session_state.stats = {
        'total': len(df), 
        'categories': df['Category'].nunique() if 'Category' in df.columns else 0
    }
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("fa-file-lines", "Total Resumes", len(df), "#6366f1"),
        ("fa-tags", "Categories", df['Category'].nunique() if 'Category' in df.columns else 0, "#8b5cf6"),
        ("fa-star", "Shortlisted", len(st.session_state.shortlisted_candidates), "#10b981"),
        ("fa-check-circle", "Processed", "Yes" if df_processed is not None else "No", "#f59e0b")
    ]
    
    for col, (icon, label, value, color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <i class="fas {icon}" style="font-size: 2rem; color: {color}; margin-bottom: 0.5rem;"></i>
                    <h3 style="margin: 0; font-size: 2rem; color: {color}; font-weight: 700;">{value}</h3>
                    <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.875rem; font-weight: 500;">{label}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    if df is not None and 'Category' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown("#### <i class='fas fa-chart-pie'></i> Category Distribution", unsafe_allow_html=True)
            
            category_counts = df['Category'].value_counts()
            colors = px.colors.sequential.Purples_r
            
            fig = go.Figure(data=[go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.6,
                marker=dict(colors=colors)
            )])
            
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="home_pie")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown("#### <i class='fas fa-chart-line'></i> Activity Timeline", unsafe_allow_html=True)
            
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            values = np.random.randint(5, 25, size=30).cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#6366f1', width=3),
                fillcolor='rgba(99, 102, 241, 0.1)'
            ))
            
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="home_line")
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# RESUME MATCHING PAGE
# ============================================================================

def render_matching_page(df_processed, extractor):
    """Render matching page with filters"""
    
    st.markdown("<h1><i class='fas fa-users-rectangle'></i> Smart Resume Matching</h1>", unsafe_allow_html=True)
    
    if df_processed is None or extractor is None:
        st.warning("Please ensure data is loaded and models are trained in Settings!")
        return
    
    # Extract experience and education if needed
    if 'experience_years' not in df_processed.columns:
        with st.spinner("Extracting experience data..."):
            df_processed['experience_years'] = df_processed.apply(
                lambda row: extract_experience_years(
                    row.get('Resume', row.get('cleaned_resume', ''))
                ), axis=1
            )
    
    if 'education_level' not in df_processed.columns:
        with st.spinner("Extracting education data..."):
            df_processed['education_level'] = df_processed.apply(
                lambda row: extract_education_level(
                    row.get('Resume', row.get('cleaned_resume', ''))
                ), axis=1
            )
    
    # Job Description Section
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### <i class='fas fa-file-contract'></i> Job Description", unsafe_allow_html=True)
    
    templates = {
        "Custom": "",
        "Senior Software Engineer": "Looking for a Senior Software Engineer with 5+ years of experience in Python, JavaScript, React, and cloud technologies. Master's degree preferred. Strong problem-solving skills required.",
        "Data Scientist": "Seeking a Data Scientist with PhD or Master's degree and 3+ years of experience in machine learning, Python, and big data technologies.",
        "Marketing Manager": "Need a Marketing Manager with Bachelor's degree and 7+ years of experience in digital marketing and team management.",
        "Junior Developer": "Looking for a Junior Developer with Bachelor's degree and 0-2 years of experience in web development."
    }
    
    template = st.selectbox(
        "Quick Templates",
        list(templates.keys()),
        key="template_select"
    )
    
    job_description = st.text_area(
        "Job Description",
        value=templates[template],
        height=150,
        placeholder="Enter job description...",
        key="job_desc_input"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filters
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown("### <i class='fas fa-filter'></i> Advanced Filters", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_experience = st.number_input(
            "Min Experience (years)",
            min_value=0, max_value=50, value=0, step=1
        )
    
    with col2:
        max_experience = st.number_input(
            "Max Experience (years)",
            min_value=0, max_value=50, value=50, step=1
        )
    
    with col3:
        education_levels = ["All", "PhD", "Master's", "Bachelor's", "Diploma", "Unknown"]
        selected_education = st.selectbox("Education Level", education_levels)
    
    with col4:
        min_match_score = st.slider("Min Match Score (%)", 0, 100, 40, 5)
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        max_results = st.number_input("Max Results", 1, 50, 10, 1)
    
    with col6:
        if 'Category' in df_processed.columns:
            categories = ["All"] + sorted(df_processed['Category'].unique().tolist())
            filter_category = st.selectbox("Category", categories)
    
    with col7:
        sort_by = st.selectbox("Sort By", ["Match Score", "Experience", "Education"])
    
    with col8:
        sort_order = st.radio("Order", ["Desc", "Asc"], horizontal=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Match Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        match_button = st.button(
            "Find Matching Candidates",
            type="primary",
            use_container_width=True,
            key="match_btn"
        )
    
    if match_button:
        if not job_description.strip():
            st.error("Please enter a job description!")
            return
        
        with st.spinner("Analyzing candidates..."):
            time.sleep(0.3)
            
            filtered_df = df_processed.copy()
            
            # Apply filters
            if 'experience_years' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['experience_years'].isna()) |
                    ((filtered_df['experience_years'] >= min_experience) &
                     (filtered_df['experience_years'] <= max_experience))
                ]
            
            if selected_education != "All" and 'education_level' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['education_level'] == selected_education]
            
            if filter_category != "All" and 'Category' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Category'] == filter_category]
            
            matcher = ResumeMatcher(extractor)
            matches = matcher.rank_candidates(filtered_df, job_description, top_n=len(filtered_df))
            matches = matches[matches['match_percentage'] >= min_match_score]
            
            # Sort
            if sort_by == "Experience" and 'experience_years' in matches.columns:
                matches = matches.sort_values('experience_years', ascending=(sort_order == "Asc"))
            elif sort_by == "Education" and 'education_level' in matches.columns:
                edu_order = {"PhD": 4, "Master's": 3, "Bachelor's": 2, "Diploma": 1, "Unknown": 0}
                matches['edu_rank'] = matches['education_level'].map(edu_order)
                matches = matches.sort_values('edu_rank', ascending=(sort_order == "Asc"))
                matches = matches.drop('edu_rank', axis=1)
            else:
                matches = matches.sort_values('match_percentage', ascending=(sort_order == "Asc"))
            
            matches = matches.head(max_results)
            st.session_state['current_matches'] = matches
            
            st.markdown(f"""
                <div class="modern-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; text-align: center;">
                    <h3 style="margin: 0; color: white;">
                        <i class="fas fa-check-circle"></i> Found {len(matches)} Matching Candidates
                    </h3>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Filtered from {len(df_processed)} total resumes</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Display Results
    if 'current_matches' in st.session_state and len(st.session_state['current_matches']) > 0:
        matches = st.session_state['current_matches']
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### <i class='fas fa-list'></i> Matched Candidates", unsafe_allow_html=True)
        
        # Export button
        col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 1])
        with col_exp2:
            export_data, export_type = export_to_excel(matches)
            if export_data:
                if export_type == "xlsx":
                    st.download_button(
                        label="Download Excel",
                        data=export_data,
                        file_name=f"matched_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_matches",
                        use_container_width=True
                    )
                else:
                    st.download_button(
                        label="Download CSV",
                        data=export_data,
                        file_name=f"matched_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_matches",
                        use_container_width=True
                    )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display candidates
        for idx, (original_idx, row) in enumerate(matches.iterrows()):
            unique_id = f"{original_idx}_{idx}"
            score = row['match_percentage']
            
            if score >= 75:
                color = "#10b981"
                badge = "Excellent"
                icon = "fa-star"
            elif score >= 60:
                color = "#6366f1"
                badge = "Good"
                icon = "fa-check"
            elif score >= 45:
                color = "#f59e0b"
                badge = "Moderate"
                icon = "fa-triangle-exclamation"
            else:
                color = "#ef4444"
                badge = "Low"
                icon = "fa-xmark"
            
            experience = row.get('experience_years', 'N/A')
            experience_text = f"{experience} years" if experience and experience != 'N/A' else "Not specified"
            education = row.get('education_level', 'Unknown')
            
            with st.expander(
                f"Candidate {idx + 1} - {row['Category']} | Match: {score:.1f}% | {badge}",
                expanded=(idx == 0)
            ):
                show_candidate_profile(row, unique_id, original_idx, idx, score, color, experience_text, education)

def show_candidate_profile(row, unique_id, original_idx, idx, score, color, experience_text, education):
    """Display candidate profile"""
    
    skills = row.get('skills', [])
    if isinstance(skills, str):
        import ast
        try:
            skills = ast.literal_eval(skills)
        except:
            skills = []
    
    # Header
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;'>
            <h3 style='margin: 0; color: white;'>
                <i class="fas fa-user"></i> Candidate #{idx + 1}
            </h3>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>ID: {original_idx}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("fa-chart-simple", "Match Score", f"{score:.1f}%", color),
        ("fa-tag", "Category", row['Category'], "#6366f1"),
        ("fa-briefcase", "Experience", experience_text, "#8b5cf6"),
        ("fa-graduation-cap", "Education", education, "#10b981")
    ]
    
    for col, (icon, label, value, metric_color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
                <div class="modern-card" style="text-align: center;">
                    <i class="fas {icon}" style="font-size: 1.5rem; color: {metric_color};"></i>
                    <div style="color: {metric_color}; font-size: 1.25rem; font-weight: 700; margin-top: 0.5rem;">{value}</div>
                    <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">{label}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Status
    if unique_id in st.session_state.shortlisted_candidates:
        st.markdown('<span class="status-badge status-success"><i class="fas fa-star"></i> Shortlisted</span>', unsafe_allow_html=True)
    elif unique_id in st.session_state.rejected_candidates:
        st.markdown('<span class="status-badge status-danger"><i class="fas fa-xmark"></i> Rejected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning"><i class="fas fa-clock"></i> Pending Review</span>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "Resume",
        "Skills",
        "Analysis"
    ])
    
    with tab1:
        st.markdown("#### <i class='fas fa-file-lines'></i> Full Resume Text", unsafe_allow_html=True)
        
        if 'Resume' in row and pd.notna(row['Resume']):
            resume_text = str(row['Resume'])
        else:
            resume_text = str(row.get('cleaned_resume', 'Resume text not available'))
        
        st.text_area("", resume_text, height=400, key=f"resume_{unique_id}", label_visibility="collapsed")
        
        st.download_button(
            "Download Resume",
            resume_text,
            file_name=f"candidate_{idx+1}_resume.txt",
            mime="text/plain",
            key=f"download_{unique_id}",
            use_container_width=True
        )
    
    with tab2:
        st.markdown("#### <i class='fas fa-code'></i> Extracted Skills", unsafe_allow_html=True)
        
        if skills:
            skills_html = " ".join([f'<span class="skill-badge">{skill}</span>' for skill in skills])
            st.markdown(skills_html, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("#### <i class='fas fa-key'></i> Top Keywords", unsafe_allow_html=True)
            
            words = re.findall(r'\b\w+\b', str(row.get('cleaned_resume', '')).lower())
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [w for w in words if w not in stop_words and len(w) > 3]
            top_words = Counter(words).most_common(12)
            
            col_kw1, col_kw2, col_kw3 = st.columns(3)
            for i, (word, count) in enumerate(top_words):
                with [col_kw1, col_kw2, col_kw3][i % 3]:
                    st.markdown(f"""
                        <div class="modern-card" style="text-align: center; padding: 0.75rem;">
                            <div style="font-weight: 600; color: #1e293b;">{word}</div>
                            <div style="color: #64748b; font-size: 0.8rem;">{count}×</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No skills extracted")
    
    with tab3:
        st.markdown("#### <i class='fas fa-chart-line'></i> Match Analysis", unsafe_allow_html=True)
        
        st.progress(score / 100)
        
        if score >= 75:
            recommendation = "HIGHLY RECOMMENDED"
            rec_text = "Excellent match for the position."
            rec_color = "#10b981"
            rec_icon = "fa-star"
        elif score >= 60:
            recommendation = "RECOMMENDED"
            rec_text = "Good match. Proceed to interview."
            rec_color = "#6366f1"
            rec_icon = "fa-check"
        elif score >= 45:
            recommendation = "CONSIDER"
            rec_text = "Moderate match. Review carefully."
            rec_color = "#f59e0b"
            rec_icon = "fa-triangle-exclamation"
        else:
            recommendation = "NOT RECOMMENDED"
            rec_text = "Low match score."
            rec_color = "#ef4444"
            rec_icon = "fa-xmark"
        
        st.markdown(f"""
            <div class="modern-card" style="border-left: 4px solid {rec_color};">
                <h4 style="color: {rec_color}; margin: 0 0 0.5rem 0;">
                    <i class="fas {rec_icon}"></i> {recommendation}
                </h4>
                <p style="margin: 0; color: #64748b;">{rec_text}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Shortlist", key=f"shortlist_{unique_id}", use_container_width=True):
            if unique_id not in st.session_state.shortlisted_candidates:
                st.session_state.shortlisted_candidates.append(unique_id)
            if unique_id in st.session_state.rejected_candidates:
                st.session_state.rejected_candidates.remove(unique_id)
            st.success("Shortlisted!")
            time.sleep(0.8)
            st.rerun()
    
    with col2:
        if st.button("Reject", key=f"reject_{unique_id}", use_container_width=True):
            if unique_id not in st.session_state.rejected_candidates:
                st.session_state.rejected_candidates.append(unique_id)
            if unique_id in st.session_state.shortlisted_candidates:
                st.session_state.shortlisted_candidates.remove(unique_id)
            st.error("Rejected")
            time.sleep(0.8)
            st.rerun()
    
    with col3:
        if st.button("Reset", key=f"reset_{unique_id}", use_container_width=True):
            if unique_id in st.session_state.shortlisted_candidates:
                st.session_state.shortlisted_candidates.remove(unique_id)
            if unique_id in st.session_state.rejected_candidates:
                st.session_state.rejected_candidates.remove(unique_id)
            st.info("Reset")
            time.sleep(0.8)
            st.rerun()

# ============================================================================
# OTHER PAGES
# ============================================================================

def render_analysis_page(df, df_processed):
    """Data analysis page"""
    st.markdown("<h1><i class='fas fa-chart-line'></i> Data Analysis</h1>", unsafe_allow_html=True)
    
    if df_processed is None:
        st.warning("Please preprocess data first!")
        return
    
    st.markdown(f"""
        <div class="modern-card">
            <h3><i class="fas fa-database"></i> Dataset Overview</h3>
            <p>Total resumes: <strong>{len(df_processed):,}</strong></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df_processed.head(50), use_container_width=True, height=600)

def render_insights_page(df_processed, extractor):
    """Category insights page"""
    st.markdown("<h1><i class='fas fa-magnifying-glass-chart'></i> Category Insights</h1>", unsafe_allow_html=True)
    
    if df_processed is None:
        st.warning("Please load data first!")
        return
    
    if 'Category' in df_processed.columns:
        categories = df_processed['Category'].value_counts()
        
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### <i class='fas fa-chart-bar'></i> Category Distribution", unsafe_allow_html=True)
        
        fig = go.Figure(data=[go.Bar(
            x=categories.index,
            y=categories.values,
            marker=dict(color=categories.values, colorscale='Purples')
        )])
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="insights_bar")
        st.markdown('</div>', unsafe_allow_html=True)

def render_settings_page(df):
    """Settings page"""
    st.markdown("<h1><i class='fas fa-gear'></i> Settings & Configuration</h1>", unsafe_allow_html=True)
    
    if df is None:
        st.error("Please upload data first!")
        return
    
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### <i class='fas fa-database'></i> Step 1: Data Preprocessing", unsafe_allow_html=True)
    
    if st.button("Preprocess Dataset", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            time.sleep(0.5)
            preprocessor = ResumePreprocessor()
            df_processed = preprocessor.process_dataframe(df)
            
            if 'Resume' in df.columns:
                df_processed['Resume'] = df['Resume'].values
            
            os.makedirs('data/processed', exist_ok=True)
            df_processed.to_csv('data/processed/processed_resumes.csv', index=False)
            
            st.success("Complete!")
            time.sleep(1)
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not os.path.exists('data/processed/processed_resumes.csv'):
        st.warning("Preprocess data first!")
        return
    
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### <i class='fas fa-brain'></i> Step 2: Model Training", unsafe_allow_html=True)
    
    if st.button("Train Models", type="primary", use_container_width=True):
        with st.spinner("Training..."):
            time.sleep(0.5)
            df_processed = pd.read_csv('data/processed/processed_resumes.csv')
            df_processed = df_processed.dropna(subset=['cleaned_resume'])
            
            extractor = FeatureExtractor(max_features=1000)
            tfidf_matrix, encoded_labels = extractor.fit_transform(
                df_processed['cleaned_resume'].tolist(),
                df_processed['Category'].tolist()
            )
            
            os.makedirs('models', exist_ok=True)
            extractor.save_vectorizer('models/tfidf_vectorizer.pkl')
            extractor.save_label_encoder('models/label_encoder.pkl')
            
            st.success("Complete!")
            st.cache_resource.clear()
            time.sleep(1)
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application"""
    
    selected_page, uploaded_file = render_sidebar()
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        default_path = 'data/raw/UpdatedResumeDataSet.csv'
        df = load_data(default_path) if os.path.exists(default_path) else None
    
    if os.path.exists('data/processed/processed_resumes.csv'):
        df_processed = pd.read_csv('data/processed/processed_resumes.csv')
        
        if df is not None and 'Resume' in df.columns and 'Resume' not in df_processed.columns:
            if len(df) == len(df_processed):
                df_processed['Resume'] = df['Resume'].values
        
        if 'skills' in df_processed.columns:
            import ast
            df_processed['skills'] = df_processed['skills'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
            )
    else:
        df_processed = None
    
    extractor = load_models()
    
    if selected_page == "Home":
        render_home_page(df, df_processed)
    elif selected_page == "Resume Matching":
        render_matching_page(df_processed, extractor)
    elif selected_page == "Data Analysis":
        render_analysis_page(df, df_processed)
    elif selected_page == "Category Insights":
        render_insights_page(df_processed, extractor)
    elif selected_page == "Settings":
        render_settings_page(df)

if __name__ == "__main__":
    main()