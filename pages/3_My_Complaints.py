from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.database import get_connection, init_db
from utils.ui import apply_global_styles, init_sidebar_language_selector, render_footer, check_citizen_access


# Initialize language selector in sidebar
init_sidebar_language_selector()

# Check citizen access
check_citizen_access()


LANG_LABELS = {
    "English": {
        "title": "My Complaints",
        "identify": "Identify yourself",
        "email_label": "Email used when filing complaints",
        "email_help": "Enter the same email address you used on the File Complaint page.",
        "info_enter_email": "Enter your email above to view complaints you have submitted.",
        "warn_no_live": "No live complaints were found in the database for this email. If you just submitted a complaint, try refreshing this page.",
        "filters": "Filters",
        "status": "Status",
        "category": "Category",
        "urgency": "Urgency",
        "search_id": "Search by Complaint ID",
        "your_complaints": "Your complaints",
        "info_no_match": "No complaints match the selected filters.",
        "showing": "Showing {n} {kind} complaints.",
        "kind_live": "live",
        "kind_synth": "synthetic",
        "view_details": "View Details",
        "download_receipt": "Download Receipt",
        "status_prefix": "Status:",
    },
    "Hindi": {
        "title": "मेरी शिकायतें",
        "identify": "पहचान दर्ज करें",
        "email_label": "शिकायत दर्ज करते समय उपयोग किया गया ईमेल",
        "email_help": "वही ईमेल दर्ज करें जो आपने शिकायत दर्ज करते समय उपयोग किया था।",
        "info_enter_email": "ऊपर अपना ईमेल दर्ज करें ताकि आपकी की गई शिकायतें दिखाई दें।",
        "warn_no_live": "इस ईमेल के लिए डेटाबेस में कोई सक्रिय शिकायत नहीं मिली। यदि आपने अभी शिकायत दर्ज की है तो पेज को पुनः लोड करें।",
        "filters": "फ़िल्टर",
        "status": "स्थिति",
        "category": "श्रेणी",
        "urgency": "तात्कालिकता",
        "search_id": "शिकायत आईडी से खोजें",
        "your_complaints": "आपकी शिकायतें",
        "info_no_match": "चुने गए फ़िल्टर के अनुसार कोई शिकायत नहीं मिली।",
        "showing": "कुल {n} {kind} शिकायतें दिखाई जा रही हैं।",
        "kind_live": "लाइव",
        "kind_synth": "सिंथेटिक",
        "view_details": "विस्तार देखें",
        "download_receipt": "रसीद डाउनलोड करें",
        "status_prefix": "स्थिति:",
    },
    "Hinglish": {
        "title": "My Complaints",
        "identify": "Apni pehchan batayen",
        "email_label": "Complaints file karte waqt use kiya gaya email",
        "email_help": "Wahi email daalein jo aapne File Complaint page par use kiya tha.",
        "info_enter_email": "Upar email daalne par aapki complaints dikhengi.",
        "warn_no_live": "Is email ke liye database mein koi live complaint nahi mili. Agar abhi complaint file ki hai to page reload karein.",
        "filters": "Filters",
        "status": "Status",
        "category": "Category",
        "urgency": "Urgency",
        "search_id": "Complaint ID se search karein",
        "your_complaints": "Aapki complaints",
        "info_no_match": "Selected filters ke liye koi complaint nahi mili.",
        "showing": "{n} {kind} complaints dikhayi ja rahi hain.",
        "kind_live": "live",
        "kind_synth": "synthetic",
        "view_details": "Details dekhein",
        "download_receipt": "Receipt download karein",
        "status_prefix": "Status:",
    },
}

apply_global_styles()
current_lang = st.session_state.get("language", "English")
L = LANG_LABELS.get(current_lang, LANG_LABELS["English"])

st.title(L["title"])

root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "civic_complaints.csv"

# Ensure DB schema exists
init_db(root)

# Check if user is logged in from session
user = st.session_state.get("user")

if not user:
    st.info("Please log in first to view your complaints. Go back to the home page and log in.")
    st.stop()

user_email = user.get("email", "")
user_name = user.get("name", "Unknown")

if not user_email:
    st.warning("No email found in your session. Please log in again.")
    st.stop()

st.subheader(f"Welcome, {user_name}!")

# Load live complaints for logged-in user (automatically)
live_df = pd.DataFrame()

with get_connection(root) as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, c.complaint_id, c.category, c.urgency, c.language, c.status,
               c.location, c.created_at, c.updated_at, c.text,
               d.name AS department_name
        FROM complaints c
        JOIN users u ON c.user_id = u.id
        LEFT JOIN departments d ON c.department_id = d.id
        WHERE u.email = ?
        ORDER BY c.created_at DESC
        """,
        (user_email,),
    )
    rows = cur.fetchall()
    if rows:
        # Get column names from cursor description
        columns = [description[0] for description in cur.description]
        live_df = pd.DataFrame(rows, columns=columns)

if live_df.empty:
    st.warning(L["warn_no_live"])

# Optional fallback: synthetic dataset view for demo/benchmarking
fallback_df = pd.DataFrame()
if live_df.empty and data_path.exists():
    fallback_df = pd.read_csv(data_path)

st.subheader(L["filters"])

if not live_df.empty:
    base_df = live_df.copy()
    status_options = ["All"] + sorted(base_df["status"].dropna().unique().tolist())
    category_options = ["All"] + sorted(base_df["category"].dropna().unique().tolist())
    urgency_options = ["All"] + sorted(base_df["urgency"].dropna().unique().tolist())
else:
    base_df = fallback_df.copy()
    if base_df.empty:
        status_options = ["All", "Registered", "In Progress", "Resolved"]
        category_options = ["All"]
        urgency_options = ["All", "Critical", "High", "Medium", "Low"]
    else:
        status_options = ["All", "Registered", "In Progress", "Resolved"]
        category_options = ["All"] + sorted(base_df["category"].dropna().unique().tolist())
        urgency_options = ["All", "Critical", "High", "Medium", "Low"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    status_filter = st.selectbox(L["status"], status_options)
with c2:
    category_filter = st.selectbox(L["category"], category_options)
with c3:
    urgency_filter = st.selectbox(L["urgency"], urgency_options)
with c4:
    search_id = st.text_input(L["search_id"])

# Prepare display DataFrame
if not live_df.empty:
    df = live_df.copy()
else:
    df = fallback_df.copy()
    if not df.empty:
        # Add synthetic status/urgency for fallback only
        status_cycle = ["Registered", "In Progress", "Resolved"]
        urgency_cycle = ["Critical", "High", "Medium", "Low"]
        df["status"] = [status_cycle[i % len(status_cycle)] for i in range(len(df))]
        df["urgency"] = [urgency_cycle[i % len(urgency_cycle)] for i in range(len(df))]

# Apply filters
if not df.empty:
    if status_filter != "All" and "status" in df.columns:
        df = df[df["status"] == status_filter]
    if category_filter != "All":
        df = df[df["category"] == category_filter]
    if urgency_filter != "All" and "urgency" in df.columns:
        df = df[df["urgency"] == urgency_filter]
    if search_id.strip():
        id_col = "complaint_id" if "complaint_id" in df.columns else "id"
        df = df[df[id_col].astype(str).str.contains(search_id.strip(), case=False)]

st.markdown("---")

st.subheader(L["your_complaints"])

if df.empty:
    st.info(L["info_no_match"])
else:
    # Simple statistics at top
    view_label = L["kind_live"] if not live_df.empty else L["kind_synth"]
    st.write(L["showing"].format(n=len(df), kind=view_label))

    id_col = "complaint_id" if "complaint_id" in df.columns else "id"

    for _, row in df.head(20).iterrows():
        with st.container():
            comp_id = row[id_col]
            category = row.get("category", "?")
            urgency = row.get("urgency", "?")
            status = row.get("status", "Registered")

            st.markdown(
                f"**ID:** {comp_id}  |  **Category:** {category}  |  **Urgency:** {urgency}"
            )
            st.write(row.get("text", "")[:200] + ("..." if len(row.get("text", "")) > 200 else ""))

            c_a, c_b = st.columns(2)
            with c_a:
                if st.button(L["view_details"], key=f"view_{comp_id}"):
                    # Hint to use the Track Complaint page for full details
                    st.session_state["track_complaint_id"] = str(comp_id)
                    st.switch_page("pages/4_Track_Complaint.py")
            with c_b:
                st.button(L["download_receipt"], key=f"receipt_{comp_id}")

            progress_map = {"Registered": 0.25, "In Progress": 0.6, "Resolved": 1.0}
            st.progress(progress_map.get(status, 0.25))
            st.caption(f"{L['status_prefix']} {status}")
            st.markdown("---")

# Shared footer
render_footer()
