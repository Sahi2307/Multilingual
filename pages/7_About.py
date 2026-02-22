from __future__ import annotations

import streamlit as st

from utils.ui import apply_global_styles, init_sidebar_language_selector, render_footer, check_citizen_access


# Initialize language selector in sidebar
init_sidebar_language_selector()

# Check citizen access
check_citizen_access()

ABOUT_LABELS = {
    "English": {
        "title": "About & Help",
        "overview": "Project overview",
        "overview_text": (
            "This project is an Explainable Multilingual Civic Complaint Resolution System "
            "built as an engineering capstone. It uses MuRIL for multilingual text "
            "understanding and XGBoost for urgency prediction, with SHAP for explainability."
        ),
        "citizens": "How to use (citizens)",
        "citizens_md": (
            "- Use the **File Complaint** page to submit a new issue in English, Hindi, or Hinglish.\n"
            "- After submission, you will see the predicted category, urgency level, queue position, and AI explanations.\n"
            "- Use the **My Complaints** and **Track Complaint** pages to review and monitor your submissions."
        ),
        "officials": "How to use (officials)",
        "officials_md": (
            "- Access the **Official Dashboard** to view incoming complaints, sorted by urgency and category.\n"
            "- Use the explanations to understand why a complaint was classified with a particular priority."
        ),
        "ai_expl": "Understanding AI explanations",
        "ai_expl_text": (
            "The system uses SHAP (SHapley Additive Explanations) to show how each word and each "
            "structured factor (such as emergency keywords or affected population) influenced the "
            "decisions. This helps make the model transparent and auditable."
        ),
        "contact": "Contact & support",
        "contact_text": "For any issues or suggestions, please contact your municipal IT support team.",
        "privacy": "Privacy & terms",
        "privacy_text": (
            "This demo uses synthetic data only. In a real deployment, personal data would be handled "
            "according to applicable privacy laws and municipal policies."
        ),
    },
    "Hindi": {
        "title": "परियोजना जानकारी और सहायता",
        "overview": "परियोजना सारांश",
        "overview_text": (
            "यह परियोजना व्याख्यात्मक बहुभाषी नागरिक शिकायत निवारण प्रणाली है। इसमें MuRIL का उपयोग "
            "बहुभाषी पाठ समझ के लिए और XGBoost का उपयोग तात्कालिकता अनुमान के लिए किया जाता है, साथ ही SHAP "
            "व्याख्या के लिए।"
        ),
        "citizens": "उपयोग कैसे करें (नागरिक)",
        "citizens_md": (
            "- **File Complaint** पेज से अंग्रेज़ी, हिंदी या हिंग्लिश में नई शिकायत दर्ज करें।\n"
            "- सबमिट करने के बाद श्रेणी, तात्कालिकता स्तर, कतार में स्थान और एआई व्याख्याएँ दिखाई देंगी।\n"
            "- **My Complaints** और **Track Complaint** पेज से अपनी शिकायतों की स्थिति देखें और मॉनिटर करें।"
        ),
        "officials": "उपयोग कैसे करें (अधिकारी)",
        "officials_md": (
            "- **Official Dashboard** से नई शिकायतें देखें, जिन्हें तात्कालिकता और श्रेणी के आधार पर क्रमबद्ध किया गया है।\n"
            "- व्याख्याओं की मदद से समझें कि किसी शिकायत को विशेष प्राथमिकता क्यों दी गई।"
        ),
        "ai_expl": "एआई व्याख्या को समझना",
        "ai_expl_text": (
            "प्रणाली SHAP (SHapley Additive Explanations) का उपयोग करती है ताकि यह दिखाया जा सके कि हर शब्द और "
            "संरचित कारक (जैसे आपातकालीन कीवर्ड या प्रभावित आबादी) ने निर्णय को कैसे प्रभावित किया। इससे मॉडल "
            "पारदर्शी और ऑडिट योग्य बनता है।"
        ),
        "contact": "संपर्क और सहायता",
        "contact_text": "किसी भी समस्या या सुझाव के लिए कृपया अपने नगर निगम की आईटी सहायता टीम से संपर्क करें।",
        "privacy": "गोपनीयता और शर्तें",
        "privacy_text": (
            "यह डेमो केवल सिंथेटिक डेटा का उपयोग करता है। वास्तविक परिनियोजन में व्यक्तिगत डेटा को लागू गोपनीयता "
            "कानूनों और नगर नीतियों के अनुसार संभाला जाएगा।"
        ),
    },
    "Hinglish": {
        "title": "About & Help",
        "overview": "Project overview",
        "overview_text": (
            "Yeh project ek Explainable Multilingual Civic Complaint Resolution System hai. MuRIL text samajhne ke "
            "liye aur XGBoost urgency predict karne ke liye use hota hai, SHAP explanations deta hai."
        ),
        "citizens": "Kaise use karein (citizens)",
        "citizens_md": (
            "- **File Complaint** page se English, Hindi ya Hinglish mein complaint file karein.\n"
            "- Submit ke baad aapko category, urgency level, queue position aur AI explanations dikhengi.\n"
            "- **My Complaints** aur **Track Complaint** pages se apni complaints track karein."
        ),
        "officials": "Kaise use karein (officials)",
        "officials_md": (
            "- **Official Dashboard** se aane wali complaints dekhein, urgency aur category ke hisaab se sort hui hui.\n"
            "- Explanations se samjhein ki complaint ko particular priority kyun di gayi."
        ),
        "ai_expl": "AI explanations ko samajhna",
        "ai_expl_text": (
            "System SHAP ka use karta hai taaki har shabd aur structured factor ka effect dikhaya ja sake, jisse model "
            "transparent aur auditable banta hai."
        ),
        "contact": "Contact & support",
        "contact_text": "Kisi bhi issue ya suggestion ke liye municipal IT support team se sampark karein.",
        "privacy": "Privacy & terms",
        "privacy_text": (
            "Yeh demo sirf synthetic data use karta hai. Real deployment mein personal data ko privacy kanoon aur municipal "
            "policies ke mutabik handle kiya jayega."
        ),
    },
}

apply_global_styles()
current_lang = st.session_state.get("language", "English")
A = ABOUT_LABELS.get(current_lang, ABOUT_LABELS["English"])

st.title(A["title"])

st.subheader(A["overview"])

st.write(A["overview_text"])

st.subheader(A["citizens"])

st.markdown(A["citizens_md"])

st.subheader(A["officials"])

st.markdown(A["officials_md"])

st.subheader(A["ai_expl"])

st.write(A["ai_expl_text"])

st.subheader(A["contact"])

st.write(A["contact_text"])

st.subheader(A["privacy"])

st.write(A["privacy_text"])

# Shared footer
render_footer()
