from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.database import get_complaint, list_status_updates_for_complaint, init_db, list_complaints_by_user
from utils.ui import render_footer

TRACK_LABELS = {
    "English": {
        "title": "Track Complaint",
        "enter_id": "Enter Complaint ID",
        "enter_help": "Example: C0001",
        "btn_search": "Search",
        "error_no_id": "Please enter a Complaint ID.",
        "info_live": "Complaint information (live DB)",
        "info_synth": "Complaint information (synthetic dataset)",
        "full_text": "Full complaint text",
        "progress_timeline": "Progress timeline",
        "no_updates": "No status updates recorded yet.",
        "queue_info_live": "Queue position and ETA were estimated when the complaint was registered. Live recalculation and display will be added when the full queue management UI is implemented.",
        "dept_updates": "Department updates",
        "dept_info_live": "Department remarks and escalation controls can be wired to the status updates and notifications tables in the next iteration.",
        "no_data": "No complaint data available (DB or CSV).",
        "no_found": "No complaint found with that ID in DB or CSV dataset.",
        "urgency_label": "Urgency (label)",
        "location_na": "Location: (not stored in synthetic data)",
        "progress": "Progress",
    },
    "Hindi": {
        "title": "शिकायत ट्रैक करें",
        "enter_id": "शिकायत आईडी दर्ज करें",
        "enter_help": "उदाहरण: C0001",
        "btn_search": "खोजें",
        "error_no_id": "कृपया शिकायत आईडी दर्ज करें।",
        "info_live": "शिकायत जानकारी (लाइव डेटाबेस)",
        "info_synth": "शिकायत जानकारी (सिंथेटिक डेटासेट)",
        "full_text": "पूरी शिकायत",
        "progress_timeline": "प्रगति समयरेखा",
        "no_updates": "अभी तक कोई स्थिति अपडेट दर्ज नहीं है।",
        "queue_info_live": "कतार स्थिति और अनुमानित समय शिकायत दर्ज करते समय निकाले गए थे। आगे के संस्करण में इन्हें लाइव अपडेट किया जाएगा।",
        "dept_updates": "विभागीय अपडेट",
        "dept_info_live": "अगले चरण में विभागीय टिप्पणियाँ और एस्केलेशन नियंत्रण यहां दिखाए जा सकते हैं।",
        "no_data": "कोई शिकायत डेटा उपलब्ध नहीं है (DB या CSV)।",
        "no_found": "DB या CSV डेटासेट में इस आईडी की कोई शिकायत नहीं मिली।",
        "urgency_label": "तात्कालिकता (लेबल)",
        "location_na": "स्थान: (सिंथेटिक डेटा में उपलब्ध नहीं)",
        "progress": "प्रगति",
    },
    "Hinglish": {
        "title": "Complaint track karein",
        "enter_id": "Complaint ID daalein",
        "enter_help": "Example: C0001",
        "btn_search": "Search",
        "error_no_id": "Please enter a Complaint ID.",
        "info_live": "Complaint information (live DB)",
        "info_synth": "Complaint information (synthetic dataset)",
        "full_text": "Full complaint text",
        "progress_timeline": "Progress timeline",
        "no_updates": "Abhi tak koi status updates nahi hain.",
        "queue_info_live": "Queue position aur ETA complaint register karte waqt estimate kiye gaye the.",
        "dept_updates": "Department updates",
        "dept_info_live": "Agle iteration mein department remarks aur escalation controls yahan dikhenge.",
        "no_data": "No complaint data available (DB or CSV).",
        "no_found": "DB ya CSV dataset mein is ID ki complaint nahi mili.",
        "urgency_label": "Urgency (label)",
        "location_na": "Location: (not stored in synthetic data)",
        "progress": "Progress",
    },
}

current_lang = st.session_state.get("language", "English")
T = TRACK_LABELS.get(current_lang, TRACK_LABELS["English"])

st.title(T["title"])

# ... (Previous imports match)

# Define root and data paths
root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "civic_complaints.csv"

# Ensure DB exists
init_db(root)

# Get current user
user = st.session_state.get("user")
if not user:
    st.warning("Please log in to track your complaints.")
    st.stop()

st.subheader("Your Complaints")
user_complaints = list_complaints_by_user(user["id"], project_root=root)

if not user_complaints:
    st.info("You have not filed any complaints yet.")
    st.stop()

# Convert to DataFrame for display
df_user = pd.DataFrame(user_complaints)
# Keep relevant columns
if not df_user.empty:
    cols = ["complaint_id", "category", "urgency", "status", "created_at"]
    # Filter to cols that exist
    show_cols = [c for c in cols if c in df_user.columns]
    
    # We allow selecting a complaint to view details
    selected_row = st.dataframe(
        df_user[show_cols],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    # Check if a row is selected (streamlit 1.35+ selection API, or fallback)
    # Actually, simpler to just use a selectbox or buttons for older streamlit versions if uncertain.
    # But since requirements said streamlit>=1.28.0, st.dataframe selection might be 1.35+.
    # Let's use a selectbox for robustness.
    
    selected_id = st.selectbox(
        "Select a complaint to view details:", 
        df_user["complaint_id"].tolist(),
        format_func=lambda x: f"{x} - {df_user[df_user['complaint_id']==x]['category'].values[0]} ({df_user[df_user['complaint_id']==x]['status'].values[0]})"
    )
    
    complaint_id = str(selected_id)
else:
    st.stop()

if complaint_id:
    # 1) Try live DB first
    db_rec = get_complaint(complaint_id.strip(), project_root=root)

    if db_rec is not None:
        st.markdown("---")
        st.subheader(T["info_live"])
        st.write(f"**Complaint ID:** {db_rec['id']}")
        st.write(f"**Category:** {db_rec['category']}")
        st.write(f"**Urgency:** {db_rec['urgency']}")
        st.write(f"**Language:** {db_rec['language']}")
        st.write(f"**Location:** {db_rec.get('location', '') or '(not specified)'}")
        st.write(f"**Status:** {db_rec.get('status', 'Unknown')}")
        st.write(f"**Created at:** {db_rec.get('created_at', '')}")
        st.write(f"**Last updated:** {db_rec.get('updated_at', '')}")

        st.markdown(f"### {T['full_text']}")
        st.write(db_rec.get("text", ""))

        st.markdown(f"### {T['progress_timeline']}")
        updates = list_status_updates_for_complaint(complaint_id.strip(), project_root=root)
        
        # Determine current stage for visual timeline
        current_status_db = db_rec.get("status", "Registered")
        
        # Map DB status to UI steps
        status_mapping = {
            "Registered": "Registered",
            "Assigned": "Service Person Allotted",
            "Service Person Allotted": "Service Person Allotted",  # Direct mapping
            "In Progress": "Service on Process",
            "Service on Process": "Service on Process",  # Direct mapping
            "Resolved": "Completed",
            "Completed": "Completed",  # Direct mapping
            "Closed": "Completed"
        }
        
        # UI Stages requested by user
        stages = ["Registered", "Service Person Allotted", "Service on Process", "Completed"]
        
        current_ui_status = status_mapping.get(current_status_db, "Registered")
        
        try:
            current_idx = stages.index(current_ui_status)
        except ValueError:
            current_idx = 0
            
        progress_cols = st.columns(len(stages))
        for i, stage in enumerate(stages):
            with progress_cols[i]:
                if i < current_idx:
                    # Completed steps
                    st.markdown(f"✅ **{stage}**")
        
                elif i == current_idx:
                    # Current step (Active)
                    st.markdown(f"🔵 **{stage}**")
                else:
                    # Future steps
                    st.markdown(f"⚪ {stage}")

        if not updates:
            st.info(T["no_updates"])
        else:
            # Deduplicate updates by keeping only the latest for each unique status
            seen_statuses = set()
            unique_updates = []
            for upd in reversed(updates):  # Reverse to keep latest
                status = upd.get("status", "")
                if status not in seen_statuses:
                    seen_statuses.add(status)
                    unique_updates.append(upd)
            unique_updates.reverse()  # Back to chronological order
            
            for upd in unique_updates:
                status = upd.get("status", "")
                remarks = upd.get("remarks", "")
                # Change display text
                display_status = "Assigned work" if status == "Service Person Allotted" else status
                st.markdown(f"- **{display_status}**")
                if remarks:
                    st.caption(f"  *Note: {remarks}*")
        
        # ... (Queue info etc)

    else:
        # Check synthetic dataset if not found in DB
        df = pd.read_csv(data_path)
        row = df[df["complaint_id"] == complaint_id.strip()].head(1)

        if row.empty:
            st.warning(T["no_found"])
        else:
            rec = row.iloc[0]
            st.markdown("---")
            st.subheader(T["info_synth"])
            st.write(f"**Complaint ID:** {rec['complaint_id']}")
            st.write(f"**Category:** {rec['category']}")
            st.write(f"**{T['urgency_label']}:** {rec['urgency']}")
            st.write(f"**Language:** {rec['language']}")
            st.write(f"**{T['location_na']}")

            st.markdown(f"### {T['full_text']}")
            st.write(rec["text"])

            st.markdown(f"### {T['progress']}")
            # Simple synthetic timeline
            steps = ["Registered", "Under Review", "In Progress", "Resolved"]
            progress_index = 2
            for i, step in enumerate(steps):
                if i <= progress_index:
                    st.markdown(f"- ✅ {step}")
                else:
                    st.markdown(f"- ☐ {step}")

            st.markdown("### Queue information")
            st.write(
                "Queue position and reasoning are not stored for the synthetic "
                "dataset. Live complaints use the queue management logic in the "
                "backend."
            )

            st.markdown("### Department updates")
            st.info(
                "Department updates and remarks will appear here once the "
                "database-backed workflow is fully wired to this view."
            )

# Shared footer
render_footer()
