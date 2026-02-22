from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import time

from utils.database import (
    get_connection,
    init_db,
    list_open_complaints_for_department,
    update_complaint_assignment,
    insert_status_update,
    get_complaint,
    get_department_id_for_category,
    execute_query,
    get_child_complaints
)
from utils.ui import (
    apply_global_styles,
    init_sidebar_language_selector,
    render_footer,
    check_official_access
)
from utils.auth import change_password

# ---------------- INIT ----------------
init_sidebar_language_selector()
check_official_access()
apply_global_styles()

current_lang = st.session_state.get("language", "English")

# ---------------- LABELS ----------------
OL = {
    "English": {
        "title": "Official Dashboard",
        "welcome": "Welcome, Official",
        "tabs": ["Overview", "Department Queue", "My Complaints", "Settings"],
        "overview": {
            "pending_dept": "Pending in Dept",
            "my_tasks": "Assigned to Me",
            "resolved_today": "Resolved Today (Me)"
        },
        "queue": {
            "header": "Department Queue (Unassigned)",
            "assign_me": "Assign to Me",
            "no_items": "No unassigned complaints in department."
        },
        "my_work": {
            "header": "My Active Cases",
            "new_status": "New Status",
            "remarks": "Remarks",
            "update_btn": "Submit Update",
            "no_items": "No active cases assigned to you."
        },
        "settings": {
            "change_pwd": "Change Password"
        }
    }
}

L = OL.get(current_lang, OL["English"])

# ---------------- HEADER ----------------
st.title(L["title"])

user = st.session_state.get("user")
if not user:
    st.error("Session expired.")
    st.stop()

dept_id = user.get("department_id")
off_id = user.get("id")

st.markdown(f"**{L['welcome']} {user['name']}** (Dept ID: {dept_id})")

# ---------------- TABS ----------------
tab_over, tab_queue, tab_my, tab_set = st.tabs(L["tabs"])

# ================= OVERVIEW TAB =================
with tab_over:
    st.header("Dashboard")

    with get_connection() as conn:
        pending_dept_count = conn.execute(
            "SELECT COUNT(*) FROM complaints "
            "WHERE department_id=? AND status NOT IN ('Resolved','Closed')",
            (dept_id,)
        ).fetchone()[0]

        assigned_me_count = conn.execute(
            "SELECT COUNT(*) FROM complaints "
            "WHERE assigned_to=? AND status NOT IN ('Resolved','Closed')",
            (off_id,)
        ).fetchone()[0]

        resolved_me_today = conn.execute(
            "SELECT COUNT(*) FROM complaints "
            "WHERE assigned_to=? AND status='Resolved' "
            "AND date(updated_at)=date('now')",
            (off_id,)
        ).fetchone()[0]

    c1, c2, c3 = st.columns(3)
    c1.metric(L["overview"]["pending_dept"], pending_dept_count)
    c2.metric(L["overview"]["my_tasks"], assigned_me_count)
    c3.metric(L["overview"]["resolved_today"], resolved_me_today)

# ================= QUEUE TAB =================
with tab_queue:
    st.subheader(L["queue"]["header"])

    with get_connection() as conn:
        q_df = pd.read_sql_query(
            """
            SELECT * FROM complaints
            WHERE department_id=?
            AND (assigned_to IS NULL OR assigned_to=0)
            AND status NOT IN ('Resolved','Closed')
            ORDER BY
                CASE urgency
                    WHEN 'Critical' THEN 4
                    WHEN 'High' THEN 3
                    WHEN 'Medium' THEN 2
                    WHEN 'Low' THEN 1
                    ELSE 0
                END DESC,
                created_at ASC
            """,
            conn,
            params=(dept_id,)
        )

    if q_df.empty:
        st.info(L["queue"]["no_items"])
    else:
        for _, row in q_df.iterrows():
            cluster_badge = ""
            if row.get("cluster_size", 1) > 1:
                cluster_badge = f" 🔥 [Cluster: {row['cluster_size']}]"

            with st.expander(
                f"{row['category']} - {row['urgency']}{cluster_badge} "
                f"({row['created_at'][:10]})"
            ):
                st.write(f"**Location:** {row['location']}")
                st.write(f"**Description:** {row['text']}")
                
                # Show child complaints if clustered
                if row.get("cluster_size", 1) > 1:
                    st.markdown("---")
                    st.markdown(f"**🔗 Linked Reports ({row['cluster_size'] - 1} others):**")
                    children = get_child_complaints(row['complaint_id'])
                    if children:
                        child_data = []
                        for child in children:
                            child_data.append({
                                "Date": child["created_at"][:10],
                                "Reporter": child.get("name", "Unknown"), # Name might not be in complaints table? Check schema.
                                # 'name' is in USERS table. 'complaints' has user_id. 
                                # But we can just show description/User ID for now.
                                "Description": child["text"]
                            })
                        st.table(child_data)
                    else:
                        st.caption("No linked records found despite cluster count.")

                if st.button(
                    L["queue"]["assign_me"],
                    key=f"assign_{row['complaint_id']}"
                ):
                    if update_complaint_assignment(row["complaint_id"], off_id):
                        st.success("Assigned successfully")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Assignment failed")

# ================= MY COMPLAINTS TAB =================
with tab_my:
    st.subheader(L["my_work"]["header"])

    with get_connection() as conn:
        my_df = pd.read_sql_query(
            """
            SELECT * FROM complaints
            WHERE assigned_to=?
            AND status NOT IN ('Resolved','Closed')
            ORDER BY
                CASE urgency
                    WHEN 'Critical' THEN 4
                    WHEN 'High' THEN 3
                    WHEN 'Medium' THEN 2
                    WHEN 'Low' THEN 1
                    ELSE 0
                END DESC
            """,
            conn,
            params=(off_id,)
        )

    if my_df.empty:
        st.info(L["my_work"]["no_items"])
    else:
        sel_id = st.selectbox(
            "Select Case",
            my_df["complaint_id"].tolist()
        )

        row = my_df[my_df["complaint_id"] == sel_id].iloc[0]

        with st.expander("Case Details", expanded=True):
            st.write(f"**Category:** {row['category']}")
            st.write(f"**Description:** {row['text']}")
            st.write(f"**Location:** {row['location']}")
            st.write(f"**Urgency:** {row['urgency']}")

        st.markdown("### Update Status")
        with st.form("status_form"):
            new_status = st.selectbox(
                L["my_work"]["new_status"],
                ["Service Person Allotted", "Service on Process", "Completed"]
            )
            remarks = st.text_area(L["my_work"]["remarks"])
            submitted = st.form_submit_button(L["my_work"]["update_btn"])

            if submitted:
                try:
                    insert_status_update(
                        sel_id,
                        new_status,
                        remarks,
                        official_id=off_id
                    )
                    st.success("Status updated")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

# ================= SETTINGS TAB =================
with tab_set:
    st.subheader(L["settings"]["change_pwd"])

    with st.form("change_pwd"):
        old_p = st.text_input("Current Password", type="password")
        new_p = st.text_input("New Password", type="password")
        conf_p = st.text_input("Confirm New Password", type="password")
        submit = st.form_submit_button("Update Password")

        if submit:
            if new_p != conf_p:
                st.error("Passwords do not match")
            else:
                success, msg = change_password(off_id, old_p, new_p)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

# ---------------- FOOTER ----------------
render_footer()
