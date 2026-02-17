# ==========================================
# STYLING MODULE - DIELLA AI
# ==========================================

import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the app."""
    
    css = """
    <style>
    /* Overall app styling */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #ec4899;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --info-color: #3b82f6;
        --dark-bg: #0f172a;
        --light-text: #e2e8f0;
        --border-color: #1e293b;
    }
    
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(to right, #6366f1, #ec4899);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: white;
        font-size: 32px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .header-subtitle {
        color: #e2e8f0;
        font-size: 14px;
        margin-top: 5px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        color: #e2e8f0;
        font-size: 28px;
        font-weight: bold;
        margin-top: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(to right, #6366f1, #ec4899);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 16px;
        background-color: #1e293b;
        border: 1px solid #334155;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to right, #6366f1, #ec4899);
        border: none;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }
    
    .stSidebar [data-testid="stSidebarUserContent"] {
        padding-top: 20px;
    }
    
    /* Input fields */
    .stTextInput input,
    .stSelectbox select,
    .stDateInput input {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
    }
    
    .stTextInput input:focus,
    .stSelectbox select:focus,
    .stDateInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 8px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Cards/Containers */
    .card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid #10b981 !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid #ef4444 !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid #f59e0b !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
    }
    
    /* Divider */
    hr {
        border-color: #334155 !important;
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }
    
    p, span, div {
        color: #cbd5e1 !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #1e293b !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #6366f1, #ec4899);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    
    /* Animation classes */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-in;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .badge-primary {
        background-color: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid #6366f1;
    }
    
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid #10b981;
    }
    
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
    }
    
    /* Charts */
    .plotly-graph-div {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #334155 !important;
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


def create_metric_card(label: str, value: str, icon: str = "📊", help_text: str = ""):
    """Create styled metric card."""
    html = f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="metric-label">{icon} {label}</div>
                <div class="metric-value">{value}</div>
            </div>
        </div>
        {f'<div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">{help_text}</div>' if help_text else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_badge(text: str, color: str = "primary"):
    """Create styled badge."""
    html = f'<span class="badge badge-{color}">{text}</span>'
    st.markdown(html, unsafe_allow_html=True)


def create_card(title: str = "", content: str = ""):
    """Create styled card container."""
    html = f"""
    <div class="card">
        {f'<h3 style="margin-top: 0;">{title}</h3>' if title else ''}
        <div>{content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)