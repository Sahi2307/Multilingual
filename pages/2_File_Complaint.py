from __future__ import annotations
import datetime as dt
import random
import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.complaint_processor import ComplaintProcessor, ProcessedComplaintResult
from utils.ui import apply_global_styles, init_sidebar_language_selector, render_footer, check_citizen_access
from utils.geo_locations import COMMON_LOCATIONS


# Initialize language selector in sidebar
init_sidebar_language_selector()

# Check citizen access
check_citizen_access()


@st.cache_resource
def get_complaint_processor() -> ComplaintProcessor:
    """Create and cache a single :class:`ComplaintProcessor` instance."""
    root = Path(__file__).resolve().parents[1]
    return ComplaintProcessor(project_root=root)


def map_affected_population(label: str) -> int:
    """Map human-readable affected population to encoded value."""
    mapping = {
        "Few individuals": 0,
        "One street / lane": 1,
        "Neighborhood / locality": 2,
        "Large area / crowd": 3,
    }
    return mapping.get(label, 1)


def highlight_keywords(text: str, keywords) -> str:
    """Return markdown-formatted text with keywords bolded.

    Args:
        text: Original complaint text.
        keywords: Iterable of keyword/token strings.

    Returns:
        Markdown string where matching keywords are wrapped in ``**``.
    """
    highlighted = text
    for kw in keywords:
        clean_kw = kw.strip()
        if not clean_kw:
            continue
        try:
            pattern = re.escape(clean_kw)
            highlighted = re.sub(
                pattern,
                f"**{clean_kw}**",
                highlighted,
                flags=re.IGNORECASE,
            )
        except re.error:
            continue
    return highlighted


LABELS = {
    "English": {
        "title": "File a Complaint",
        "details": "Complaint details",
        "registered": "Complaint registered",
        "ai_expl": "AI explanations",
        "why_cat": "Why categorized as this?",
        "why_pri": "Why this priority?",
        "lang_label": "Language / भाषा / Language",
        "name": "Name",
        "email": "Email",
        "phone": "Phone",
        "location": "Location",
        "category_hint": "Category (optional hint)",
        "complaint_language": "Complaint language",
        "affected_population": "Affected population",
        "complaint_description": "Complaint description",
        "complaint_help": "Describe the issue in as much detail as possible.",
        "upload_photos": "Upload photos (optional, up to 3)",
        "submit": "Submit complaint",
        "error_no_text": "Please enter a complaint description before submitting.",
        "spinner_main": "Running MuRIL analysis, ML models, and saving your complaint...",
        "spinner_shap": "Generating explainable AI insights with SHAP...",
    },
    "Hindi": {
        "title": "शिकायत दर्ज करें",
        "details": "शिकायत विवरण",
        "registered": "शिकायत दर्ज हो गई है",
        "ai_expl": "एआई व्याख्या",
        "why_cat": "इस श्रेणी में क्यों रखा गया?",
        "why_pri": "यह प्राथमिकता क्यों?",
        "lang_label": "भाषा / Language",
        "name": "नाम",
        "email": "ईमेल",
        "phone": "फ़ोन",
        "location": "स्थान",
        "category_hint": "श्रेणी (वैकल्पिक संकेत)",
        "complaint_language": "शिकायत की भाषा",
        "affected_population": "प्रभावित आबादी",
        "complaint_description": "शिकायत का विवरण",
        "complaint_help": "समस्या को जितना हो सके विस्तार से लिखें।",
        "upload_photos": "फोटो अपलोड करें (वैकल्पिक, अधिकतम 3)",
        "submit": "शिकायत दर्ज करें",
        "error_no_text": "कृपया शिकायत का विवरण लिखने के बाद ही सबमिट करें।",
        "spinner_main": "MuRIL विश्लेषण, मॉडल और शिकायत सहेजी जा रही है...",
        "spinner_shap": "SHAP के माध्यम से व्याख्यात्मक एआई जानकारी तैयार की जा रही है...",
    },
    "Hinglish": {
        "title": "Complaint file karein",
        "details": "Complaint details",
        "registered": "Complaint register ho gayi hai",
        "ai_expl": "AI explanations",
        "why_cat": "Is category mein kyun?",
        "why_pri": "Yeh priority kyun?",
        "lang_label": "Language / भाषा / Language",
        "name": "Naam",
        "email": "Email",
        "phone": "Phone",
        "location": "Location",
        "category_hint": "Category (optional hint)",
        "complaint_language": "Complaint ki language",
        "affected_population": "Affected population",
        "complaint_description": "Complaint description",
        "complaint_help": "Issue ko detail mein describe karein.",
        "upload_photos": "Photos upload karein (optional, max 3)",
        "submit": "Complaint submit karein",
        "error_no_text": "Complaint description likhne ke baad hi submit karein.",
        "spinner_main": "MuRIL analysis, ML models aur complaint save ho rahi hai...",
        "spinner_shap": "SHAP se explainable AI insights ban rahe hain...",
    },
}

# Keep language in sync with global setting
current_lang = st.session_state.get("language", "English")
labels = LABELS.get(current_lang, LABELS["English"])

# Global look & feel + light page-specific styling
apply_global_styles()
st.markdown(
    """
    <style>
    .file-section-title {
        font-weight: 600;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
    /* Ensure no white backgrounds in dark mode */
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: transparent !important;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<h1 class='file-section-title'>{labels['title']}</h1>", unsafe_allow_html=True)

st.markdown("---")

st.subheader(labels["details"])

# Custom CSS to reduce spacing and gaps
st.markdown("""
<style>
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
    .stTextInput, .stSelectbox, .stTextArea {
        margin-bottom: -10px;
    }
</style>
""", unsafe_allow_html=True)

with st.container(border=True):
    # 1. Category Selection
    ui_category = st.selectbox(
        labels["category_hint"],
        ["Let AI decide", "Sanitation", "Water Supply", "Transportation"],
    )

    # 2. Geolocation & Search UI
    st.markdown("<p style='margin-bottom: -15px; font-weight: bold;'>Location</p>", unsafe_allow_html=True)
    col_loc_in, col_loc_btn = st.columns([4, 1], vertical_alignment="bottom")

    with col_loc_in:
        search_query = st.text_input(
            "Location Search", 
            placeholder="Type address or area...", 
            key="loc_search",
            help="Type address to see suggestions like Google Maps.",
            label_visibility="collapsed"
        )

    with col_loc_btn:
        try:
            from streamlit_js_eval import get_geolocation
            loc_data = get_geolocation()
            # Custom styled button container to ensure it matches precisely
            if st.button("📍Location", help="Ask permission & fetch current GPS address"):

                if loc_data:
                    lat = loc_data.get('coords', {}).get('latitude')
                    lon = loc_data.get('coords', {}).get('longitude')
                    try:
                        from geopy.geocoders import Nominatim
                        geolocator = Nominatim(user_agent="civic_complaint_system_v2")
                        location = geolocator.reverse((lat, lon), language='en', timeout=3)
                        if location:
                            st.session_state["loc_search"] = location.address
                            st.rerun()
                    except Exception as e:
                        if "getaddrinfo" in str(e) or "Max retries exceeded" in str(e):
                            st.error("📡 Network Error: Could not reach the location service. Please check your internet connection or type the address manually.")
                        else:
                            st.error(f"Location Error: {e}")
                else:
                    st.warning("Please allow GPS access.")
        except ImportError:
            pass

    # 3. Autocomplete Suggestions (India-prioritized)
    location_input = search_query
    if search_query:
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="civic_complaint_resolution_v2")
            # Limit results to India and reduce timeout for faster feel
            locations = geolocator.geocode(
                search_query, 
                exactly_one=False, 
                limit=6, 
                timeout=2,
                country_codes='in'
            )
            if locations:
                options = [loc.address for loc in locations]
                # Label is "Suggestions" for a better feel
                selected = st.selectbox("📍 Suggestions (Select to confirm)", options=options, key="suggestion_box")
                location_input = selected
        except Exception:
            pass



    # 4. Remaining Details
    affected_label = st.selectbox(
        labels["affected_population"],
        [
            "Few individuals",
            "One street / lane",
            "Neighborhood / locality",
            "Large area / crowd",
        ],
    )

    complaint_text = st.text_area(
        labels["complaint_description"],
        height=180,
        max_chars=2000,
        help=labels["complaint_help"],
    )

    st.write("") # Spacer
    submitted = st.button(labels["submit"], type="primary", use_container_width=True)

if submitted:

    if not complaint_text.strip() or not location_input.strip():
        st.error("Please fill in all mandatory fields (Location and Description).")
        st.stop()

    # Get current user info from session
    user_data = st.session_state.get("user", {})
    user_name = user_data.get("name", "Anonymous")
    user_email = user_data.get("email", "")
    user_phone = user_data.get("phone", "")
    
    # Use language from session state
    complaint_language = st.session_state.get("language", "English")

    processor = get_complaint_processor()

    with st.spinner(labels["spinner_main"]):
        result: ProcessedComplaintResult = processor.process_complaint(
            name=user_name,
            email=user_email,
            phone=user_phone,
            location=location_input if location_input.strip() else "Not specified",
            complaint_text=complaint_text,
            language=complaint_language,
            affected_label=affected_label,
            category_hint=None if ui_category == "Let AI decide" else ui_category,
        )

    # Short spinner to reflect SHAP explanation generation (already done in pipeline)
    with st.spinner(labels["spinner_shap"]):
        pass

    complaint_id = result.complaint_id
    cat_exp = result.category_explanation
    urg_exp = result.urgency_explanation
    department = result.department_name
    queue_position = result.queue_position
    eta_text = result.eta_text

    st.markdown("---")
    st.subheader(labels["registered"])

    c_info, c_meta = st.columns([2, 2])
    with c_info:
        st.write(f"**Complaint ID:** {complaint_id}")
        st.write(f"**Detected category:** {cat_exp.predicted_label} ({cat_exp.confidence * 100:.1f}% confidence)")
        st.write(f"**Assigned urgency:** {urg_exp.predicted_label} ({urg_exp.confidence * 100:.1f}% confidence)")
    with c_meta:
        st.write(f"**Department routing:** {department}")
        st.write(f"**Queue position:** #{queue_position}")
        st.write(f"**Estimated response time:** {eta_text}")

    st.markdown("---")
    st.subheader(labels["ai_expl"])

    # Category Explanation — simple paragraph
    with st.expander(labels["why_cat"], expanded=True):
        top_tokens = [kw.strip() for kw in cat_exp.top_keywords[:5] if kw.strip()]
        if top_tokens:
            keyword_str = ", ".join(f"**{kw}**" for kw in top_tokens)
            st.markdown(
                f"Your complaint has been categorized as **{cat_exp.predicted_label}** "
                f"with **{cat_exp.confidence * 100:.1f}%** confidence. "
                f"The AI model identified key words such as {keyword_str} in your description, "
                f"which are strongly associated with **{cat_exp.predicted_label}** issues. "
                f"These terms helped the model distinguish this complaint from other categories."
            )
        else:
            st.markdown(
                f"Your complaint has been categorized as **{cat_exp.predicted_label}** "
                f"with **{cat_exp.confidence * 100:.1f}%** confidence. "
                f"The AI model analyzed the overall context and language of your description "
                f"to determine this is a **{cat_exp.predicted_label}** issue."
            )

    # Urgency Explanation — template paragraph based on category + urgency + scope
    with st.expander(labels["why_pri"], expanded=True):
        urgency = urg_exp.predicted_label
        category = cat_exp.predicted_label
        affected = urg_exp.structured_features.get("affected_population", 0)
        has_emergency_kw = urg_exp.structured_features.get("emergency_keyword_score", 0) > 0

        # Scope description based on affected_population (0=few, 1=street, 2=neighborhood, 3=large area)
        if affected >= 3:
            scope_desc = "a large area or crowd"
            scope_impact = "The widespread impact across a large population significantly raised the urgency"
        elif affected >= 2:
            scope_desc = "an entire neighborhood or locality"
            scope_impact = "The fact that this affects an entire neighborhood contributed to a higher urgency level"
        elif affected >= 1:
            scope_desc = "a street or lane"
            scope_impact = "Since this impacts an entire street, the urgency has been raised accordingly"
        else:
            scope_desc = "a few individuals"
            scope_impact = "As this currently affects a small number of individuals, the urgency reflects the limited scope"

        # Category-specific context
        cat_context = {
            "Sanitation": "Sanitation issues, if left unresolved, can lead to health hazards, spread of diseases, and unhygienic living conditions for residents.",
            "Water Supply": "Water supply disruptions directly impact daily life, hygiene, and can pose serious health risks especially for vulnerable populations.",
            "Transportation": "Transportation and road infrastructure issues can cause accidents, traffic disruptions, and affect commuter safety on a daily basis.",
        }
        cat_reason = cat_context.get(category, "This type of civic issue requires timely attention from the concerned department.")

        # Urgency-specific opening
        if urgency == "Critical":
            urgency_opening = (
                f"This complaint has been assigned **Critical** priority because it describes "
                f"a severe **{category}** issue affecting **{scope_desc}**."
            )
        elif urgency == "High":
            urgency_opening = (
                f"This complaint has been assigned **High** priority because it reports "
                f"a significant **{category}** problem impacting **{scope_desc}**."
            )
        elif urgency == "Medium":
            urgency_opening = (
                f"This complaint has been assigned **Medium** priority. The reported "
                f"**{category}** issue affecting **{scope_desc}** requires attention but "
                f"is not immediately life-threatening."
            )
        else:  # Low
            urgency_opening = (
                f"This complaint has been assigned **Low** priority. The reported "
                f"**{category}** concern affecting **{scope_desc}** is noted and will be "
                f"addressed in the standard resolution timeline."
            )

        # Emergency keyword addition
        kw_note = ""
        if has_emergency_kw:
            kw_note = " Additionally, the AI detected **emergency-related keywords** in the description, which further elevated the priority."

        # Final 2-3 line paragraph
        st.markdown(
            f"{urgency_opening} {cat_reason} "
            f"{scope_impact}.{kw_note}"
        )

# Shared footer
render_footer()
