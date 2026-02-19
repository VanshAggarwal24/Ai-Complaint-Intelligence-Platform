import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Complaint Intelligence Platform",
    layout="wide"
)

# ================= GLOBAL FONT STYLING =================
st.markdown("""
<style>

/* Increase overall readability */
html, body, [class*="css"]  {
    font-size: 18px !important;
}

/* Subtitle styling */
.subtitle {
    font-size: 20px !important;
    color: #cbd5e1;
}

/* Section headings */
h2 {
    font-size: 26px !important;
}

/* Labels */
strong {
    font-size: 19px !important;
}

/* Text area font */
textarea {
    font-size: 17px !important;
}

</style>
""", unsafe_allow_html=True)

# ================= DOWNLOAD STOPWORDS =================
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# ================= LOAD MODEL =================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ================= TEXT CLEANING =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ================= BUSINESS RULES =================
def business_action(category, confidence):

    # 🔎 If confidence is low → send for manual review
    if confidence < 0.6:
        return (
            "Manual Review Team",
            "Low",
            "Low confidence prediction. Requires human validation."
        )

    # Otherwise apply normal routing logic
    if "Credit" in category:
        return "Compliance Team", "High", "Escalate to credit bureau liaison."
    elif "Debt" in category:
        return "Legal & Recovery", "High", "Review for legal risk and customer dispute."
    elif "Mortgage" in category:
        return "Lending Department", "Medium", "Verify loan records and repayment schedule."
    elif "Student loan" in category:
        return "Loan Support Team", "Medium", "Review repayment assistance eligibility."
    else:
        return "Customer Support", "Low", "Standard customer handling procedure."


# ================= UI =================

st.title("AI Complaint Intelligence Platform")
st.markdown('<p class="subtitle">AI-powered complaint classification and intelligent business routing system.</p>', unsafe_allow_html=True)

st.markdown("---")

# ================= INPUT =================
st.header("Enter Customer Complaint")

user_input = st.text_area("", height=200)
st.caption(f"Character Count: {len(user_input)}")

if st.button("Analyze & Route Complaint"):

    if user_input.strip() == "":
        st.warning("Please enter complaint text.")
    else:
        with st.spinner("Analyzing complaint using AI model..."):
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]
            confidence = np.max(probabilities)
            if confidence < 0.6:
                st.warning("⚠ Low confidence prediction. Consider manual review.")


            department, priority, action = business_action(prediction, confidence)

        st.markdown("---")
        st.header("AI Classification Result")

        # ===== CATEGORY CARD =====
        st.markdown(
            f"""
            <div style="
                background-color:#1f2937;
                padding:20px;
                border-radius:12px;
                margin-bottom:15px;">
                <h4 style="color:white;">Category</h4>
                <p style="color:#38bdf8; font-size:22px;">{prediction}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ===== CONFIDENCE CARD =====
        st.markdown(
            f"""
            <div style="
                background-color:#1f2937;
                padding:20px;
                border-radius:12px;
                margin-bottom:15px;">
                <h4 style="color:white;">Confidence</h4>
                <p style="color:#4ade80; font-size:22px;">{round(confidence*100,2)} %</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ===== PRIORITY CARD =====
        priority_color = "#ef4444" if priority=="High" else "#facc15" if priority=="Medium" else "#4ade80"

        st.markdown(
            f"""
            <div style="
                background-color:#1f2937;
                padding:20px;
                border-radius:12px;
                margin-bottom:15px;">
                <h4 style="color:white;">Priority Level</h4>
                <p style="color:{priority_color}; font-size:22px;">{priority}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ===== BUSINESS ROUTING =====
        st.markdown("---")
        st.header("Business Routing Recommendation")

        st.markdown(f"""
        <div style="font-size:20px; margin-bottom:10px;">
        <b>Assigned Department:</b> {department}
        </div>
        <div style="font-size:20px;">
        <b>Recommended Action:</b> {action}
        </div>
        """, unsafe_allow_html=True)

        # ===== RISK SCORE =====
        base_score = confidence * 100

        if priority == "High":
            risk_score = min(base_score * 1.3, 100)
        elif priority == "Medium":
            risk_score = min(base_score * 1.1, 100)
        else:
            risk_score = base_score

        risk_score = round(risk_score, 2)

        st.markdown("---")
        st.header("Risk Score")

        st.progress(risk_score / 100)
        st.write(f"Risk Score: {risk_score}/100")

        # ===== MODEL CONFIDENCE GRAPH =====
        with st.expander("View Model Confidence Breakdown"):

            import pandas as pd
            import matplotlib.pyplot as plt

            classes = model.classes_

            prob_df = pd.DataFrame({
                "Category": classes,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(prob_df["Category"], prob_df["Probability"])
            ax.set_xlabel("Probability")
            ax.set_ylabel("Category")
            ax.set_title("Model Confidence by Category")

            st.pyplot(fig)

        # ===== AUTO RESPONSE =====
        st.markdown("---")
        st.header("Suggested Customer Response")

        response = f"""
Dear Customer,

Thank you for bringing your concern regarding {prediction} to our attention.
Our {department} has been assigned to review your complaint.

We are currently investigating the issue and will provide an update shortly.

Best Regards,
Customer Resolution Team
"""

        st.text_area("Generated Response", response, height=200)

st.markdown("---")
st.caption("Enterprise AI Complaint Intelligence Platform")
