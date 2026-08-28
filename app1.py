
import streamlit as st
import numpy as np
import joblib

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = joblib.load("best_classification_model.pkl")

# ---------------------------------------------------------
# FEATURE NAMES
# ---------------------------------------------------------
feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",

    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",

    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #888888;
    margin-bottom: 25px;
}

/* Metric cards */
.metric-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    text-align: center;
}

.metric-value {
    font-size: 27px;
    font-weight: 700;
}

.metric-label {
    font-size: 14px;
    color: #888888;
}

/* Section headers */
.section-header {
    font-size: 22px;
    font-weight: 650;
    margin-top: 15px;
    margin-bottom: 10px;
}

/* Prediction box */
.prediction-box {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.3);
    text-align: center;
    margin-top: 20px;
}

.prediction-title {
    font-size: 18px;
    color: #888888;
}

.prediction-result {
    font-size: 32px;
    font-weight: 700;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:

    st.title("🔬 About")

    st.write(
        "This application uses a machine learning model "
        "to classify breast cancer samples based on "
        "30 numerical features."
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("Balanced Logistic Regression")

    st.divider()

    st.subheader("📊 Model Performance")

    st.write("Accuracy: **97.37%**")
    st.write("Precision: **97.22%**")
    st.write("Recall: **98.59%**")
    st.write("F1 Score: **97.90%**")
    st.write("ROC-AUC: **99.77%**")

    st.divider()

    st.caption(
        "Machine Learning Classification Project"
    )

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🔬 Breast Cancer Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter the measurements below and use the trained model '
    'to generate a classification prediction.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">97.37%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">97.22%</div>
        <div class="metric-label">Precision</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">98.59%</div>
        <div class="metric-label">Recall</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">97.90%</div>
        <div class="metric-label">F1 Score</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">99.77%</div>
        <div class="metric-label">ROC-AUC</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------
st.markdown(
    '<div class="section-header">📋 Patient Measurements</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the 30 numerical features required by the trained model."
)

# ---------------------------------------------------------
# MEAN FEATURES
# ---------------------------------------------------------
with st.expander("📏 Mean Measurements", expanded=True):

    mean_features = feature_names[:10]

    mean_inputs = []

    col1, col2, col3 = st.columns(3)

    columns = [col1, col2, col3]

    for i, feature in enumerate(mean_features):

        with columns[i % 3]:

            value = st.number_input(
                feature.title(),
                value=0.0,
                format="%.6f",
                key=f"mean_{i}"
            )

            mean_inputs.append(value)

# ---------------------------------------------------------
# ERROR FEATURES
# ---------------------------------------------------------
with st.expander("📐 Measurement Errors", expanded=True):

    error_features = feature_names[10:20]

    error_inputs = []

    col1, col2, col3 = st.columns(3)

    columns = [col1, col2, col3]

    for i, feature in enumerate(error_features):

        with columns[i % 3]:

            value = st.number_input(
                feature.title(),
                value=0.0,
                format="%.6f",
                key=f"error_{i}"
            )

            error_inputs.append(value)

# ---------------------------------------------------------
# WORST FEATURES
# ---------------------------------------------------------
with st.expander("📊 Worst Measurements", expanded=True):

    worst_features = feature_names[20:30]

    worst_inputs = []

    col1, col2, col3 = st.columns(3)

    columns = [col1, col2, col3]

    for i, feature in enumerate(worst_features):

        with columns[i % 3]:

            value = st.number_input(
                feature.title(),
                value=0.0,
                format="%.6f",
                key=f"worst_{i}"
            )

            worst_inputs.append(value)

# ---------------------------------------------------------
# COMBINE FEATURES
# ---------------------------------------------------------
features = mean_inputs + error_inputs + worst_inputs

# ---------------------------------------------------------
# BUTTONS
# ---------------------------------------------------------
st.divider()

button_col1, button_col2, button_col3 = st.columns([1, 1, 3])

with button_col1:

    predict_button = st.button(
        "🔍 Predict",
        use_container_width=True
    )

with button_col2:

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True
    )

# ---------------------------------------------------------
# RESET
# ---------------------------------------------------------
if reset_button:

    st.rerun()

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if predict_button:

    input_data = np.array(features).reshape(1, -1)

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    st.divider()

    st.subheader("🎯 Prediction Result")

    # Class 0
    if prediction == 0:

        st.success(
            "Prediction: **Class 0**"
        )

    # Class 1
    else:

        st.success(
            "Prediction: **Class 1**"
        )

    # -----------------------------------------------------
    # PROBABILITIES
    # -----------------------------------------------------

    st.subheader("📊 Prediction Probabilities")

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:

        st.metric(
            "Class 0 Probability",
            f"{probabilities[0] * 100:.2f}%"
        )

        st.progress(
            float(probabilities[0])
        )

    with prob_col2:

        st.metric(
            "Class 1 Probability",
            f"{probabilities[1] * 100:.2f}%"
        )

        st.progress(
            float(probabilities[1])
        )

    st.caption(
        "Probabilities represent the model's estimated confidence "
        "for each class."
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "Built with Python • Scikit-learn • Streamlit"
)
