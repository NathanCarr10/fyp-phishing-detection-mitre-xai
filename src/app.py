# src/app.py
#
# Streamlit demo app for phishing detection + MITRE mapping + XAI explanation.
#
# Features:
# - Paste email text into a text area
# - Classify as phishing or legitimate
# - Show phishing probability with confidence visualization
# - Show MITRE ATT&CK mapping with attack technique details
# - Show top XAI features with explanations of their relevance
# - Display generated analysis figures from simulation results
# - Customizable threshold and feature counts
# - Email input validation and helpful error messages
# - Example emails for quick testing
#
# Run from project root:
#   streamlit run src/app.py

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

# Ensure src/ is importable when Streamlit runs from project root
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mvp_baseline import load_model
from mvp_baseline import NB_MODEL_PATH, MODEL_PATH
from xai_explainer import explain_email
from attacker_sim import mitre_mapping
from utils import classify_email as classify_email_utils


# ================== PATHS ================== #

FIGURES_DIR = PROJECT_ROOT / "simulation_results" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"


# ================== FEATURE EXPLANATIONS ================== #

FEATURE_EXPLANATIONS = {
    "urgent": "Creates time pressure to bypass rational decision-making",
    "verify": "Requests credential validation, a common phishing tactic",
    "confirm": "Asks for sensitive information confirmation",
    "click": "Encourages clicking potentially malicious links",
    "account": "References user accounts to appear legitimate",
    "security": "Fake security alerts are a common phishing technique",
    "alert": "Alarm words trigger immediate action without scrutiny",
    "payment": "Financial terms attract attention and create urgency",
    "action": "Action-oriented language increases click-through rates",
    "suspend": "Account suspension threats are highly effective",
    "password": "Password requests are the hallmark of phishing",
    "confirm": "Confirmation requests for sensitive data",
    "update": "False update prompts are frequently used",
    "error": "Technical errors create urgency to fix issues",
    "warning": "Warning messages trigger immediate responses",
}


# ================== PAGE CONFIG ================== #

# ================== PAGE CONFIG ================== #

st.set_page_config(
    page_title="Phishing Detection with XAI & MITRE Mapping",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .confidence-bar {
        background: linear-gradient(90deg, #ff4444 0%, #ffaa00 50%, #44aa44 100%);
        height: 8px;
        border-radius: 4px;
        margin: 5px 0;
    }
    .feature-importance {
        padding: 10px;
        border-left: 4px solid #0066cc;
        background-color: #f0f5ff;
        margin: 8px 0;
        border-radius: 4px;
    }
    .mitre-box {
        padding: 12px;
        background-color: #fff3cd;
        border-left: 4px solid #ff9800;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ================== CACHED MODEL LOADER ================== #

@st.cache_resource
def get_model(model_choice: str = "logistic_regression"):
    """
    Load and cache the vectorizer and classifier.
    
    Args:
        model_choice: "logistic_regression" or "naive_bayes"
    
    Returns:
        tuple: (vectorizer, classifier) or None if files not found
        
    Raises:
        FileNotFoundError: If model files are missing
        Exception: For other loading errors
    """
    try:
        # Select model path based on choice
        if model_choice == "naive_bayes":
            model_path = NB_MODEL_PATH
            model_name = "Multinomial Naive Bayes"
        else:
            model_path = MODEL_PATH
            model_name = "Logistic Regression"
        
        vectorizer, clf = load_model(model_path=model_path)
        return vectorizer, clf
    except FileNotFoundError as e:
        st.error(
            f"❌ **Model Files Not Found**\n\n"
            f"The trained model files are missing. Please run the following command first:\n\n"
            f"```\npython src/mvp_baseline.py\n```\n\n"
            f"This will train both the Logistic Regression and Naive Bayes models and save them to `models/`.\n\n"
            f"Error: {str(e)}"
        )
        st.stop()
    except Exception as e:
        st.error(
            f"❌ **Error Loading Model**\n\n"
            f"An unexpected error occurred while loading the model:\n\n"
            f"```\n{str(e)}\n```\n\n"
            f"Please check that all required files are in place and try again."
        )
        st.stop()


# ================== HELPER FUNCTIONS ================== #

def validate_email_input(text: str) -> tuple[bool, str]:
    """
    Validate email input and provide helpful feedback.
    
    Returns:
        tuple: (is_valid, message)
    """
    if not text or not text.strip():
        return False, "Email text cannot be empty."
    
    if len(text.strip()) < 5:
        return False, "Email text is too short. Please provide at least 5 characters."
    
    if len(text.strip()) > 10000:
        return False, "Email text is too long. Maximum 10,000 characters allowed."
    
    return True, "Email text is valid."


def get_confidence_color(prob: float) -> str:
    """Get color based on confidence probability."""
    if prob >= 0.75:
        return "red"
    elif prob >= 0.55:
        return "orange"
    elif prob >= 0.40:
        return "gray"
    else:
        return "green"


def get_confidence_label(prob: float) -> str:
    """Get confidence label based on probability."""
    if prob >= 0.85:
        return "Very High Confidence"
    elif prob >= 0.70:
        return "High Confidence"
    elif prob >= 0.55:
        return "Moderate Confidence"
    elif prob >= 0.40:
        return "Low Confidence"
    else:
        return "Very Low Confidence"


def safe_explain_email(email_text: str, num_features: int, threshold: float, use_lime: bool) -> dict:
    """
    Safely explain an email with error handling.
    
    Returns:
        dict: Explanation result or error info
    """
    try:
        return explain_email(
            email_text,
            num_features=num_features,
            threshold=threshold,
            use_lime=use_lime,
        )
    except Exception as e:
        return {
            "method": f"⚠️ Explanation Error: {str(e)}",
            "top_features": [],
        }


# ================== SIDEBAR ================== #

st.sidebar.title("⚙️ Settings & Help")

st.sidebar.subheader("Model Selection")

model_choice = st.sidebar.radio(
    "Choose Classification Model:",
    options=["logistic_regression", "naive_bayes"],
    format_func=lambda x: "🔵 Logistic Regression" if x == "logistic_regression" else "🟠 Multinomial Naive Bayes",
    help="Compare predictions from different ML algorithms. Both use TF-IDF features.",
)

st.sidebar.divider()

st.sidebar.subheader("Prediction Settings")

threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Higher threshold = fewer false positives (but may miss some phishing). "
         "Lower threshold = catch more phishing (but more false positives).",
)

num_features = st.sidebar.slider(
    "Number of XAI Features to Show",
    min_value=3,
    max_value=15,
    value=10,
    step=1,
    help="Number of important keywords shown in the explanation. More features provide deeper insight.",
)

use_lime = st.sidebar.checkbox(
    "Use LIME for Explanations",
    value=True,
    help="LIME provides local interpretable explanations. Falls back to linear weights if unavailable.",
)

show_figures = st.sidebar.checkbox(
    "Show Simulation Figures",
    value=True,
    help="Display generated experiment charts and analysis results.",
)

st.sidebar.divider()
st.sidebar.subheader("📖 Quick Help")

with st.sidebar.expander("How does this work?", expanded=False):
    st.markdown("""
    **This dashboard:**
    1. **Classifies** emails as phishing or legitimate using ML
    2. **Explains** which keywords influenced the decision (XAI)
    3. **Maps** detected phishing techniques to MITRE ATT&CK framework
    4. **Shows** experiment results comparing defense strategies
    
    **Model Details:**
    - **Logistic Regression**: Linear classifier, good for interpretability
    - **Multinomial Naive Bayes**: Probabilistic classifier, often effective for text
    - Features: TF-IDF (Term Frequency-Inverse Document Frequency)
    - Training Data: 1,000+ phishing & legitimate emails
    - XAI Method: LIME (Local Interpretable Model-agnostic Explanations)
    
    **Pro Tip:** Compare both models to see how different algorithms handle the same email!
    """)

with st.sidebar.expander("What is MITRE ATT&CK?", expanded=False):
    st.markdown("""
    MITRE ATT&CK is a knowledge base of adversary tactics and techniques 
    based on real-world observations.
    
    **Phishing-related techniques detected:**
    - **T1566.002**: Phishing - Spearphishing Link
    - **T1566.001**: Phishing - Spearphishing Attachment
    - **T1598**: Phishing for Information
    - And others based on email content
    """)

with st.sidebar.expander("Understanding confidence levels", expanded=False):
    st.markdown("""
    **Very High (85%+):** Strong evidence of phishing
    **High (70-85%):** Likely phishing email
    **Moderate (55-70%):** Borderline, manual review advised
    **Low (40-55%):** Likely legitimate
    **Very Low (<40%):** Strong evidence of legitimacy
    """)

st.sidebar.divider()
st.sidebar.info("💡 Tip: Use different thresholds to balance false positives vs false negatives.")


# ================== HEADER ================== #

st.title("🔒 Phishing Detection with XAI & MITRE Mapping")

st.markdown("""
Paste an email below to:
- ✅ Detect if it's phishing or legitimate
- 📊 See the confidence score and key indicators
- 🗺️ Map to MITRE ATT&CK phishing techniques
- 🔍 Understand *why* (explainable AI)
""")

st.divider()


# ================== EXAMPLE EMAILS ================== #

example_emails = {
    "⚠️ Account Suspension (Phishing)": (
        "URGENT: Your account may be closed soon. "
        "We have detected suspicious activity on your account. "
        "Please respond within 24 hours to avoid losing access. "
        "Click here to resolve this issue: http://secure-verification-example.com/login"
    ),
    "💰 Wire Transfer Urgency (Phishing)": (
        "Dear Client,\n\n"
        "I need your urgent assistance with a confidential transaction. "
        "A sum of $2.5 million needs to be transferred immediately. "
        "For your discretion and security, please respond with your bank details. "
        "This is a legitimate business opportunity. "
        "Time-sensitive: respond within 24 hours."
    ),
    "🏦 Fake Bank Alert (Phishing)": (
        "Dear Customer,\n\n"
        "We detected unusual activity on your account. "
        "Please verify your details immediately by clicking the link below: "
        "https://bank-security-verify.com/confirm-account "
        "Failure to verify will result in account suspension."
    ),
    "✓ Meeting Confirmation (Legitimate)": (
        "Hi Team,\n\n"
        "Our meeting scheduled for Friday at 2 PM has been confirmed. "
        "The agenda includes quarterly planning and project updates. "
        "Please bring your reports. Looking forward to seeing everyone.\n\n"
        "Best regards,\nJohn"
    ),
    "📅 Project Update (Legitimate)": (
        "Hi David,\n\n"
        "Just confirming that the project deliverables will be ready by the end of week. "
        "I've completed the testing phase and everything looks good. "
        "Let me know if you need anything else.\n\n"
        "Thanks,\nSarah"
    ),
}

st.subheader("📋 Example Emails")
col1, col2 = st.columns(2)

with col1:
    example_choice = st.selectbox(
        "Quick Load Example",
        options=list(example_emails.keys()),
        label_visibility="collapsed",
    )

with col2:
    if st.button("📋 Load Example", key="load_example"):
        st.session_state.email_text = example_emails[example_choice]


# ================== MAIN INPUT ================== #

st.subheader("📧 Email Input")

default_email = example_emails["⚠️ Account Suspension (Phishing)"]

if "email_text" not in st.session_state:
    st.session_state.email_text = default_email

email_text = st.text_area(
    "Paste email content below (plain text only):",
    value=st.session_state.email_text,
    height=200,
    placeholder="Paste full email header and body here...",
)

st.session_state.email_text = email_text

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    analyse_clicked = st.button("🔍 Analyse Email", type="primary", use_container_width=True)

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.email_text = ""
        st.rerun()

with col3:
    if st.button("📋 Load Phishing", use_container_width=True):
        st.session_state.email_text = example_emails["⚠️ Account Suspension (Phishing)"]
        st.rerun()


# ================== MAIN ANALYSIS ================== #

if analyse_clicked:
    # Validate input
    is_valid, validation_msg = validate_email_input(email_text)
    
    if not is_valid:
        st.warning(f"⚠️ {validation_msg}")
    else:
        try:
            # Load model based on sidebar selection
            vectorizer, clf = get_model(model_choice=model_choice)
            
            # Classify email
            pred_label, phishing_prob = classify_email_utils(
                vectorizer,
                clf,
                email_text,
                threshold=threshold,
            )
            
            # Get MITRE mapping
            mitre_label = mitre_mapping(email_text)
            
            # Get explanation
            explanation = safe_explain_email(
                email_text,
                num_features=num_features,
                threshold=threshold,
                use_lime=use_lime,
            )
            
            prediction_text = "🚨 Phishing" if pred_label == 1 else "✅ Legitimate"
            confidence_label = get_confidence_label(phishing_prob)
            
            # ===== RESULT CARDS =====
            st.divider()
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            # Prediction card
            with col1:
                st.markdown("### Prediction")
                if pred_label == 1:
                    st.error(f"### 🚨 **PHISHING**", icon="⚠️")
                else:
                    st.success(f"### ✅ **LEGITIMATE**", icon="✓")
            
            # Probability card
            with col2:
                st.markdown("### Confidence")
                
                # Display probability as percentage
                prob_percentage = phishing_prob * 100
                st.metric(
                    label="Phishing Probability",
                    value=f"{prob_percentage:.1f}%",
                    delta=confidence_label,
                )
                
                # Confidence bar visualization
                st.markdown(f'<div class="confidence-bar"></div>', unsafe_allow_html=True)
            
            # MITRE card
            with col3:
                st.markdown("### MITRE ATT&CK")
                st.markdown(f'<div class="mitre-box"><strong>{mitre_label}</strong></div>', 
                           unsafe_allow_html=True)
            
            st.divider()
            
            # ===== DETAILED EXPLANATION =====
            st.subheader("🔍 Why This Classification?")
            
            st.info(f"**Explanation Method:** {explanation['method']}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📌 Top Important Keywords")
                
                top_features = explanation.get("top_features", [])
                
                if top_features:
                    for i, feature in enumerate(top_features, 1):
                        term = feature.get("term", "").lower()
                        weight = feature.get("weight", 0.0)
                        
                        # Get explanation if available
                        explanation_text = FEATURE_EXPLANATIONS.get(term, 
                            "This keyword influences the classification.")
                        
                        # Color code based on weight (negative = legit, positive = phishing)
                        if weight > 0:
                            color = "🔴"  # Phishing indicator
                        elif weight < 0:
                            color = "🟢"  # Legitimate indicator
                        else:
                            color = "⚪"  # Neutral
                        
                        st.markdown(f'''
                        <div class="feature-importance">
                        <strong>{i}. {term.upper()}</strong> {color}<br>
                        <small>Weight: {weight:+.4f}</small><br>
                        <small>💡 {explanation_text}</small>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.warning("❌ No explanation features were available for this email. "
                             "This could mean the email is borderline or uses unusual language.")
            
            with col2:
                st.subheader("📈 Decision Threshold")
                st.metric(
                    label="Current Threshold",
                    value=f"{threshold:.2f}",
                )
                
                if pred_label == 1:
                    st.success(f"✅ {phishing_prob:.3f} > {threshold:.2f}")
                else:
                    st.info(f"ℹ️ {phishing_prob:.3f} < {threshold:.2f}")
                
                st.caption("📌 Adjust threshold in sidebar to change sensitivity")
            
            st.divider()
            
            # ===== INTERPRETATION GUIDE =====
            st.subheader("📚 Understanding the Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🔴 Phishing Indicators:**
                - Urgent/threatening language
                - Requests for credentials or sensitive info
                - Suspicious links or attachments
                - Spoofed sender addresses
                - Emotional manipulation tactics
                """)
            
            with col2:
                st.markdown("""
                **🟢 Legitimate Indicators:**
                - Clear business context
                - No credential requests
                - Professional tone
                - Known sender information
                - Specific, relevant content
                """)
        
        except FileNotFoundError:
            st.error(
                "❌ **Model Files Not Found**\n\n"
                "The trained model files are missing from `models/`. "
                "Please run the training script first:\n\n"
                "```bash\npython src/mvp_baseline.py\n```"
            )
        except ValueError as e:
            st.error(
                f"❌ **Invalid Email Format**\n\n"
                f"Error: {str(e)}\n\n"
                f"Please ensure the email text is in a valid format."
            )
        except Exception as e:
            st.error(
                f"❌ **Analysis Error**\n\n"
                f"An unexpected error occurred: {str(e)}\n\n"
                f"**Debugging Info:**\n"
                f"- Email length: {len(email_text)} characters\n"
                f"- Threshold: {threshold}\n\n"
                f"Please try again or contact support."
            )


# ================== SIMULATION RESULTS & FIGURES ================== #

if show_figures:
    st.divider()
    st.header("📈 Experiment Results & Figures")
    
    st.info(
        "These figures show the performance of the phishing detection system "
        "against adversarial attacks with different thresholds and rule combinations."
    )
    
    figure_files = [
        ("Detection Rate by Threshold", FIGURES_DIR / "threshold_detection_rate.png"),
        ("Bypass Rate by Threshold", FIGURES_DIR / "threshold_bypass_rate.png"),
        ("F1 Score by Threshold", FIGURES_DIR / "threshold_f1_score.png"),
        ("Accuracy by Threshold", FIGURES_DIR / "threshold_accuracy.png"),
        ("Detection Rate by Attack Type", FIGURES_DIR / "attack_type_detection_rate.png"),
        ("Detection Rate by Rule Chain", FIGURES_DIR / "rule_chain_detection_rate.png"),
        ("Detection Rate by MITRE Technique", FIGURES_DIR / "mitre_detection_rate.png"),
    ]

    existing_figures = [(title, path) for title, path in figure_files if path.exists()]

    if not existing_figures:
        st.warning(
            "📊 No experiment figures found yet.\n\n"
            "To generate the figures, please run:\n\n"
            "```bash\n"
            "python src/analyse_simulation_results.py\n"
            "python src/visualise_results.py\n"
            "```\n\n"
            "This will analyze the simulation results and create visualization charts."
        )
    else:
        st.success(f"✅ Found {len(existing_figures)} figure(s) from experiments")
        
        # Display figures in a grid
        for title, fig_path in existing_figures:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("📊")
                st.caption(title)
            
            with col2:
                try:
                    st.image(str(fig_path), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image: {str(e)}")


# ================== FOOTER ================== #

st.divider()
st.markdown("""
### 📚 Learn More
- **README.md** - Project overview and usage guide
- **ARCHITECTURE.md** - Technical system design
- **PROJECT_REPORT.md** - Research methodology and results
- **DEVELOPMENT.md** - Developer documentation

### 🔗 References
- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [LIME Explainability](https://github.com/marcotcr/lime)
- [Phishing Detection Research](https://en.wikipedia.org/wiki/Phishing)

---
**Phishing Detection System v1.0** | Built with Python, scikit-learn, and Streamlit
""")