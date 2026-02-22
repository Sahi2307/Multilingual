from __future__ import annotations

from pathlib import Path
from datetime import datetime
import streamlit as st
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import pandas as pd

from utils.auth import (
    register_user,
    login_user,
    validate_password_strength,
    validate_email,
    validate_phone,
)
from utils.session_manager import init_session_state, is_authenticated, clear_session, set_user_session, show_user_info
from utils.ui import apply_global_styles, init_sidebar_language_selector, render_footer
from utils.database import initialize_database
import os


@st.cache_resource
def init_db() -> None:
    """Initialize database once on first run."""
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'civic_complaints.db')
    if not os.path.exists(db_path):
        initialize_database()


# Initialize database if needed
init_db()


def init_language() -> None:
    """Initialise the global language setting in ``st.session_state``."""
    if "language" not in st.session_state:
        st.session_state["language"] = "English"


def load_stats() -> tuple[int, int, float]:
    """Load simple statistics from the synthetic dataset.

    Returns ``(total_complaints, resolved_today, avg_resolution_days)``.
    Values are heuristic placeholders based on the CSV file.
    """

    root = Path(__file__).resolve().parent
    data_path = root / "data" / "civic_complaints.csv"
    if not data_path.exists():
        return 0, 0, 0.0

    df = pd.read_csv(data_path)
    total = len(df)
    resolved_today = int(total * 0.1)
    avg_resolution_days = 2.3
    return total, resolved_today, avg_resolution_days


# Simple multilingual label dictionary for the landing page
LABELS = {
    "English": {
        "app_title": "Explainable Multilingual Civic Complaint System",
        "overview": "Today's overview",
        "how_it_works": "How it works",
        "quick_actions": "Quick actions",
        "tagline": (
            "A simple, explainable portal for citizens to file complaints in "
            "English, Hindi, or Hinglish and track them through municipal offices."
        ),
        "step1_title": "1. Submit complaint",
        "step1_desc": "Citizens submit complaints in English, Hindi, or Hinglish with optional photos.",
        "step2_title": "2. AI classification",
        "step2_desc": "MuRIL and custom ML models analyze the complaint and predict category & urgency with explanations.",
        "step3_title": "3. Routing & tracking",
        "step3_desc": "The system routes complaints to the right department and lets citizens track status and queue position.",
        "file_button": "File Complaint",
        "track_button": "Track Complaint",
        "official_button": "Official Dashboard",
        "login": "Login",
        "register": "New User? Register Here",
        "email": "Email/Username",
        "password": "Password",
        "role": "Role",
        "login_btn": "Login",
        "forgot_password": "Forgot Password?",
    },
    "Hindi": {
        "app_title": "व्याख्यात्मक बहुभाषी नागरिक शिकायत प्रणाली",
        "overview": "आज का सारांश",
        "how_it_works": "कैसे काम करता है",
        "quick_actions": "त्वरित कार्रवाइयाँ",
        "tagline": (
            "एक सरल, व्याख्यात्मक पोर्टल जहाँ नागरिक अंग्रेज़ी, हिंदी या हिंग्लिश में "
            "शिकायत दर्ज कर सकते हैं और उन्हें नगर निगम कार्यालयों तक ट्रैक कर सकते हैं।"
        ),
        "step1_title": "1. शिकायत दर्ज करें",
        "step1_desc": "नागरिक अंग्रेज़ी, हिंदी या हिंग्लिश में शिकायत और चाहें तो फोटो के साथ दर्ज करते हैं।",
        "step2_title": "2. एआई वर्गीकरण",
        "step2_desc": "MuRIL और अन्य मॉडल शिकायत की श्रेणी और तात्कालिकता का अनुमान लगाकर व्याख्या दिखाते हैं।",
        "step3_title": "3. रूटिंग और ट्रैकिंग",
        "step3_desc": "प्रणाली शिकायत को सही विभाग तक भेजती है और नागरिक को स्थिति व कतार स्थान दिखाती है।",
        "file_button": "शिकायत दर्ज करें",
        "track_button": "शिकायत ट्रैक करें",
        "official_button": "अधिकारी डैशबोर्ड",
        "login": "लॉगिन",
        "register": "नए उपयोगकर्ता? यहां पंजीकरण करें",
        "email": "ईमेल/उपयोगकर्ता नाम",
        "password": "पासवर्ड",
        "role": "भूमिका",
        "login_btn": "लॉगिन करें",
        "forgot_password": "पासवर्ड भूल गए?",
    },
    "Hinglish": {
        "app_title": "Explainable Multilingual Civic Complaint System",
        "overview": "Aaj ka overview",
        "how_it_works": "Kaise kaam karta hai",
        "quick_actions": "Quick actions",
        "tagline": (
            "Ek simple portal jahan citizens English, Hindi ya Hinglish mein complaint "
            "file karke usse track kar sakte hain."
        ),
        "step1_title": "1. Complaint file karein",
        "step1_desc": "Citizens English, Hindi ya Hinglish mein complaint aur photos ke saath bhejte hain.",
        "step2_title": "2. AI classification",
        "step2_desc": "MuRIL aur ML models complaint ki category aur urgency predict karke explanation dete hain.",
        "step3_title": "3. Routing & tracking",
        "step3_desc": "System complaint ko sahi department tak bhejkar status aur queue position dikhata hai.",
        "file_button": "Complaint file karein",
        "track_button": "Complaint track karein",
        "official_button": "Official dashboard",
        "login": "Login",
        "register": "Naye user? Yahan register karein",
        "email": "Email/Username",
        "password": "Password",
        "role": "Role",
        "login_btn": "Login karein",
        "forgot_password": "Password bhool gaye?",
    },
}


# Shared styling, language, and session state
apply_global_styles()
init_language()
init_session_state()

# Initialize global language selector in sidebar
init_sidebar_language_selector()

root = Path(__file__).resolve().parent

current_lang = st.session_state.get("language", "English")
labels = LABELS.get(current_lang, LABELS["English"])

# ---- Hero header ----
st.markdown(
    f"<h1 style='margin-bottom:0.25rem'>{labels['app_title']}</h1>",
    unsafe_allow_html=True,
)
st.write(labels["tagline"])

# User info in top right
col_user = st.columns([3])[0]
with col_user:
    if is_authenticated():
        st.success(
            f"Logged in as {st.session_state.user_name} ({st.session_state.user_role.capitalize()})",
        )
        if st.button("Logout", key="logout_btn"):
            clear_session()
            st.rerun()

st.markdown("---")



# Rate limiting tracking
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "captcha_required" not in st.session_state:
    st.session_state.captcha_required = False

# ---- Login & registration section ----
st.subheader("Login")

with st.form("login_form"):
    login_col1, login_col2 = st.columns(2)
    with login_col1:
        login_email = st.text_input(labels["email"])
        login_password = st.text_input(labels["password"], type="password")
    with login_col2:
        login_role = st.selectbox(
            labels["role"],
            options=["citizen", "official", "admin"],
            format_func=lambda x: x.capitalize(),
        )
        st.markdown(f"[{labels['forgot_password']}](#)")

    # CAPTCHA after 3 failed attempts
    if st.session_state.captcha_required:
        captcha_input = st.text_input("🤖 CAPTCHA: What is 7 + 8?")

    login_submitted = st.form_submit_button(labels["login_btn"], use_container_width=True)

if login_submitted:
    # Validate inputs
    if not login_email or not login_password:
        st.error("❌ Please enter both email and password")
    elif st.session_state.captcha_required and captcha_input != "15":
        st.error("❌ Incorrect CAPTCHA answer")
    else:
        # Attempt login
        captcha_solved = st.session_state.captcha_required and captcha_input == "15"
        
        success, message, user_data = login_user(
            login_email.strip(), 
            login_password.strip(), 
            login_role,
            captcha_passed=captcha_solved
        )

        if not success:
            st.session_state.login_attempts += 1
            if st.session_state.login_attempts >= 3:
                if not st.session_state.captcha_required:
                    st.session_state.captcha_required = True
                    st.rerun()
            st.error(f"❌ {message}")
        else:
            # Reset login attempts
            st.session_state.login_attempts = 0
            st.session_state.captcha_required = False

            # Create session via set_user_session
            set_user_session(user_data)
            st.success(f"✅ {message}")

            # Redirect based on role
            if user_data["role"] == "citizen":
                st.switch_page("pages/2_File_Complaint.py")
            elif user_data["role"] == "official":
                st.switch_page("pages/5_Official_Dashboard.py")
            else:  # admin
                st.switch_page("pages/6_Admin_Panel.py")

st.markdown("---")

# ---- Registration section ----
st.subheader(f"📝 {labels['register']}")

with st.expander("Open registration form"):
    with st.form("registration_form"):
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            r_name = st.text_input("Full Name *")
            r_email = st.text_input("Email *")
            r_phone = st.text_input("Phone Number (e.g., +919876543210)")
            r_location = st.text_input("Location/Ward")
        with r_col2:
            r_role = st.selectbox(
                "Role *",
                options=["citizen", "official", "admin"],
                index=0,
                help="Officials require admin approval. Admin self-registration is for demo only.",
            )
            if r_role == "admin":
                st.warning("⚠️ Admin accounts should be created by existing admins in production.")
            r_password = st.text_input("Password *", type="password")
            r_confirm = st.text_input("Confirm Password *", type="password")

        register_submitted = st.form_submit_button("Create Account", use_container_width=True)

    if register_submitted:
        # Validation
        if not all([r_name, r_email, r_password, r_confirm]):
            st.error("❌ Please fill all required fields")
        elif r_password != r_confirm:
            st.error("❌ Passwords do not match")
        elif not validate_email(r_email):
            st.error("❌ Invalid email format")
        elif r_phone and not validate_phone(r_phone):
            st.error("❌ Invalid phone format. Use +91XXXXXXXXXX")
        else:
            # Validate password strength
            is_valid, pwd_msg = validate_password_strength(r_password)
            if not is_valid:
                st.error(f"❌ {pwd_msg}")
            else:
                # Register user
                success, message, user_id = register_user(
                    name=r_name.strip(),
                    email=r_email.strip(),
                    phone=r_phone.strip() if r_phone else None,
                    password=r_password,
                    role=r_role,
                    location=r_location.strip() if r_location else None,
                )

                if success:
                    st.success(f"✅ {message}")
                    st.info("Please login with your credentials")
                else:
                    st.error(f"❌ {message}")

st.markdown("---")

# ---- Quick Stats Section (public) ----
st.subheader("📊 Quick Stats")

# For demo purposes, use the fixed numbers from the specification
fixed_total = 1245
fixed_resolved_today = 87
fixed_avg_resolution = 3.2
active_departments = 5
system_uptime = 99.8

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total complaints registered", value=fixed_total)
with c2:
    st.metric("Complaints resolved today", value=fixed_resolved_today)
with c3:
    st.metric("Avg. resolution time (days)", value=f"{fixed_avg_resolution:.1f}")
with c4:
    st.metric("Active departments", value=active_departments)
with c5:
    st.metric("System uptime (%)", value=f"{system_uptime:.1f}")

st.markdown("---")

# ---- How It Works ----
st.subheader("🎯 How It Works")

steps = [
    ("📝", labels["step1_title"], labels["step1_desc"]),
    ("🤖", labels["step2_title"], labels["step2_desc"]),
    ("📊", labels["step3_title"], labels["step3_desc"]),
]

cols = st.columns(3)
for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown("<div class='civic-card'>", unsafe_allow_html=True)
        st.markdown(f"**{icon} {title}**")
        st.write(desc)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---- Feature Highlights ----
st.subheader("✨ Feature Highlights")

feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    st.markdown(
        """
    - ✨ **AI-Powered Categorization** (≥94% accuracy)
    - 🔍 **Explainable AI** with SHAP
    - 🌐 **Multilingual Support** (English, Hindi, Hinglish)
    """
    )

with feat_col2:
    st.markdown(
        """
    - 📈 **Real-time Queue Transparency**
    - 🚀 **Fast Processing** (<4 seconds)
    - 🔒 **Secure & Role-Based Access**
    """
    )

st.markdown("---")

# ---- Demo Mode ----
st.subheader("🎮 Try Demo (Guest Access)")
st.info("Experience the system without registration. Limited to viewing demo data.")

if st.button("🚀 Launch Demo Mode", use_container_width=True):
    # Create demo session
    demo_user = {
        "id": -1,
        "name": "Demo Guest",
        "email": "demo@guest.local",
        "role": "citizen",
        "is_active": True,
        "department_id": None,
    }
    set_user_session(demo_user, "demo_session_token")
    st.session_state.demo_mode = True
    st.success("✅ Demo mode activated! Redirecting...")
    st.switch_page("pages/2_File_Complaint.py")

st.markdown("---")

# ---- Footer / Contact ----
st.subheader("📞 Contact & Support")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown(
        """
    **Email:** support@civiccomplaints.gov  
    **Phone:** 1800-XXX-XXXX (24/7)  
    **Address:** Municipal IT Centre, Civic Bhawan
    """
    )

with footer_col2:
    st.markdown(
        """
    **Quick Links**  
    - [Privacy Policy](#)
    - [Terms of Service](#)
    - [About Us](#)
    """
    )

with footer_col3:
    st.markdown(
        """
    **Social Media**  
    - 🐦 Twitter: @CivicComplaints
    - 📘 Facebook: /CivicComplaints
    - 📸 Instagram: @civic_complaints
    """
    )

st.markdown("---")
st.caption("© 2024 Civic Complaint Resolution System | Built with ❤️ using Streamlit, MuRIL & XGBoost")
