import streamlit as st
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("best_classification_model.pkl")


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Breast Cancer Classification")
st.write(
    "Enter the 30 diagnostic feature values below to get a prediction "
    "from the trained machine learning model."
)

st.info(
    "The model predicts whether the sample is classified as "
    "**Malignant** or **Benign**."
)


# ============================================================
# FEATURE NAMES
# ============================================================

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


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Enter Feature Values")

inputs = []

# Create 3 columns instead of one long column
columns = st.columns(3)

for i, feature in enumerate(feature_names):

    with columns[i % 3]:

        value = st.number_input(
            feature,
            value=0.0,
            format="%.6f",
            key=feature
        )

        inputs.append(value)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔍 Predict",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Convert inputs into NumPy array
    input_data = np.array(inputs).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_data)[0]


    # ========================================================
    # CONVERT CLASS NUMBER TO ACTUAL NAME
    # ========================================================

    if prediction == 0:
        predicted_class = "Malignant"
    else:
        predicted_class = "Benign"


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("Prediction Result")

    if prediction == 0:
        st.error(
            "⚠️ Prediction: Malignant"
        )

    else:
        st.success(
            "✅ Prediction: Benign"
        )


    # ========================================================
    # DISPLAY PROBABILITIES
    # ========================================================

    st.subheader("Prediction Probabilities")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Malignant Probability",
            f"{probabilities[0] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Benign Probability",
            f"{probabilities[1] * 100:.2f}%"
        )


    # ========================================================
    # PROGRESS BARS
    # ========================================================

    st.write("### Probability Breakdown")

    st.write(
        f"Malignant: {probabilities[0] * 100:.2f}%"
    )

    st.progress(float(probabilities[0]))

    st.write(
        f"Benign: {probabilities[1] * 100:.2f}%"
    )

    st.progress(float(probabilities[1]))


# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Educational machine learning project. "
    "This application is not a medical diagnostic tool "
    "and should not be used for medical decisions."
)