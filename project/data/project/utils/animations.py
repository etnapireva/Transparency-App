# utils/animations.py

import streamlit as st
import time

def loading_spinner(message: str = "Loading..."):
    """Show animated loading spinner."""
    with st.spinner(f"⏳ {message}"):
        time.sleep(0.5)  # Simulates work
        return True


def animated_metric(label: str, value: str, icon: str = "📊"):
    """Display animated metric."""
    html = f"""
    <div style="
        animation: slideIn 0.5s ease-in;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    ">
        <div style="font-size: 24px;">{icon}</div>
        <div style="color: #94a3b8; font-size: 12px; margin-top: 8px;">{label}</div>
        <div style="color: #e2e8f0; font-size: 28px; font-weight: bold; margin-top: 8px;">{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def progress_bar_animated(value: float, label: str = ""):
    """Display animated progress bar."""
    percentage = min(100, max(0, int(value)))
    
    html = f"""
    <div style="margin: 10px 0;">
        {f'<span style="color: #94a3b8; font-size: 12px;">{label}</span>' if label else ''}
        <div style="
            background-color: #1e293b;
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin-top: 4px;
        ">
            <div style="
                background: linear-gradient(to right, #6366f1, #ec4899);
                height: 100%;
                width: {percentage}%;
                border-radius: 8px;
                transition: width 0.5s ease;
            "></div>
        </div>
        <span style="color: #94a3b8; font-size: 12px; float: right; margin-top: 4px;">{percentage}%</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)