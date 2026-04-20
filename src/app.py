# src/app.py
#
# Main Streamlit app for my phishing detection project.
#
# This app lets you paste email text or load a .eml file,
# then shows the prediction, MITRE mapping, and explanation results.
#
# Run from project root:
#   streamlit run src/app.py
#   or use .\run_app.ps1 in PowerShell

import sys
from pathlib import Path

import streamlit as st

# Ensure src/ is importable when Streamlit runs from project root
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mvp_baseline import load_model
from mvp_baseline import NB_MODEL_PATH, MODEL_PATH
from mvp_baseline import get_model_compatibility_warning
from email_ingestion import parse_eml_file, analyze_combined_text


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
    """Load the chosen model and keep it cached for reuse."""
    try:
        # Select model path based on choice
        if model_choice == "naive_bayes":
            model_path = NB_MODEL_PATH
            model_name = "Multinomial Naive Bayes"
        else:
            model_path = MODEL_PATH
            model_name = "Logistic Regression"
        
        vectorizer, clf = load_model(model_path=model_path)
        warning_text = get_model_compatibility_warning(vectorizer, clf)
        if warning_text:
            st.warning(warning_text)
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
    """Check whether the email text looks valid enough to process."""
    cleaned_text = text.strip() if isinstance(text, str) else ""

    if not cleaned_text:
        return False, "Email text cannot be empty."

    text_len = len(cleaned_text)
    if text_len < 5:
        return False, "Email text is too short. Please provide at least 5 characters."

    if text_len > 10000:
        return False, "Email text is too long. Maximum 10,000 characters allowed."

    return True, "Email text is valid."


CONFIDENCE_LABELS = [
    (0.85, "Very High Confidence"),
    (0.70, "High Confidence"),
    (0.55, "Moderate Confidence"),
    (0.40, "Low Confidence"),
    (0.00, "Very Low Confidence"),
]

CONFIDENCE_COLORS = [
    (0.75, "red"),
    (0.55, "orange"),
    (0.40, "gray"),
    (0.00, "green"),
]


def get_confidence_color(prob: float) -> str:
    """Pick a colour name based on the probability."""
    for threshold_value, color in CONFIDENCE_COLORS:
        if prob >= threshold_value:
            return color
    return "green"


def get_confidence_label(prob: float) -> str:
    """Return a simple confidence label for the score."""
    for threshold_value, label in CONFIDENCE_LABELS:
        if prob >= threshold_value:
            return label
    return "Very Low Confidence"


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

use_shap = st.sidebar.checkbox(
    "Use SHAP for Explanations (Experimental)",
    value=False,
    help="Uses SHAP values with the linear model. If SHAP is unavailable, the app falls back to linear weights.",
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
    - XAI Methods: SHAP (experimental), LIME, or linear fallback
    
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
Provide email content below to:
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
        selected_key = example_choice if example_choice is not None else "⚠️ Account Suspension (Phishing)"
        st.session_state.email_text = example_emails[selected_key]


# ================== MAIN INPUT ================== #

st.subheader("📧 Email Input")

input_mode = st.radio(
    "Input Mode",
    options=["Manual Text", "Upload .eml File"],
    horizontal=True,
    help="Use manual text entry or upload a local .eml message file.",
)

default_email = example_emails["⚠️ Account Suspension (Phishing)"]

if "email_text" not in st.session_state:
    st.session_state.email_text = default_email

analysis_text = ""
parsed_email_data = None
analyse_clicked = False

if input_mode == "Manual Text":
    email_text = st.text_area(
        "Paste email content below (plain text only):",
        value=st.session_state.email_text,
        height=200,
        placeholder="Paste full email header and body here...",
    )

    st.session_state.email_text = email_text
    analysis_text = email_text

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
else:
    uploaded_file = st.file_uploader(
        "Upload a local .eml file",
        type=["eml"],
        help="Parses email headers/body safely. Attachments are listed by filename only.",
    )

    if uploaded_file is not None:
        try:
            parsed_email_data = parse_eml_file(uploaded_file.getvalue())
            analysis_text = parsed_email_data["combined_text"]

            st.markdown("### 📄 Parsed Email Metadata")
            meta_col1, meta_col2 = st.columns(2)

            with meta_col1:
                st.text_input("From", value=parsed_email_data.get("from", ""), disabled=True)
                st.text_input("To", value=parsed_email_data.get("to", ""), disabled=True)

            with meta_col2:
                st.text_input("Subject", value=parsed_email_data.get("subject", ""), disabled=True)
                st.text_input("Date", value=parsed_email_data.get("date", ""), disabled=True)

            attachment_names = parsed_email_data.get("attachment_names", [])
            if attachment_names:
                st.caption("Attachments (filenames only):")
                st.write(", ".join(attachment_names))
            else:
                st.caption("Attachments: none detected")

            st.text_area(
                "Extracted Body (safe text)",
                value=parsed_email_data.get("body", ""),
                height=180,
                disabled=True,
            )

            analyse_clicked = st.button("🔍 Analyse Uploaded Email", type="primary", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Could not parse .eml file: {str(e)}")
            analyse_clicked = False
    else:
        st.info("Upload a `.eml` file to parse and analyze it.")

# ================== MAIN ANALYSIS ================== #

if analyse_clicked:
    # Validate input
    is_valid, validation_msg = validate_email_input(analysis_text)
    
    if not is_valid:
        st.warning(f"⚠️ {validation_msg}")
    else:
        try:
            # Load model based on sidebar selection
            selected_model = model_choice if model_choice is not None else "logistic_regression"
            vectorizer, clf = get_model(model_choice=selected_model)

            # Shared analysis flow for manual text and uploaded .eml content.
            analysis_result = analyze_combined_text(
                analysis_text,
                threshold=threshold,
                num_features=num_features,
                use_lime=use_lime,
                use_shap=use_shap,
                vectorizer=vectorizer,
                clf=clf,
            )

            pred_label = analysis_result["predicted_label"]
            phishing_prob = analysis_result["phishing_probability"]
            mitre_label = analysis_result["mitre_mapping"]
            explanation = analysis_result["xai_explanation"]
            
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
                    st.success(f"### ✅ **LEGITIMATE**", icon="✅")
            
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
                f"- Email length: {len(analysis_text)} characters\n"
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
    st.caption(
        "These charts come from the saved simulation runs, so they stay the same "
        "even if you change the model in the sidebar. The model selector only "
        "changes the live email analysis above."
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
                    st.image(str(fig_path), use_column_width=True)
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