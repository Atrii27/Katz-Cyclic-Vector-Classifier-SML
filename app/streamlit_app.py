# Import Streamlit for web app UI
import streamlit as st
# Set Streamlit page configuration
st.set_page_config(page_title="Katz Cyclic Vector Classifier (n=3)", layout="wide")
# Import joblib for loading ML models
import joblib
# Import os for file path operations
import os
# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical operations
import numpy as np
# Import sympy for symbolic mathematics
from sympy import Matrix, symbols
# Import sys for system-specific parameters and functions
import sys
# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import custom symbolic and utility functions
from src.symbolic_katz import get_katz_derivatives, check_cyclicity
from src.utils import parse_matrix_input, parse_vector_input, flatten_matrix, make_dataframe_arrow_safe
from src.features import add_matrix_features
# Load the pre-trained ML model from disk
model_path = r"C:/Users/DELL/Desktop/SRM2025IISERM/katz_cyclic_vector_ml/models/trained_model.pkl"
model = joblib.load(model_path)
# Show a sidebar message indicating the model is loaded
st.sidebar.success(" Loaded ML model for n = 3")

# Display the app title and description using HTML
st.markdown("""
    <h1 style='text-align: center; color: teal;'> Katz Cyclic Vector Classifier (n = 3)</h1>
    <p style='text-align: center;'>Compare symbolic cyclicity with ML prediction</p>
""", unsafe_allow_html=True)
# Set matrix/vector size n
n = 3
# Define symbolic variable t
t_sym = symbols('t')
# Sidebar slider for evaluation point t
t_val = st.sidebar.slider("Evaluation point t", min_value=-5.0, max_value=5.0, value=1.0, step=0.5)
# Sidebar slider for shift value a
a_val = st.sidebar.slider("Shift value a", min_value=-5.0, max_value=5.0, value=1.0, step=0.5)
# Default input for matrix D
default_D = "1,1,0;0,1,1;0,0,1"
# Default input for basis vectors
default_basis = "1,0,0;0,1,0;0,0,1"

# Create two columns for matrix and basis input
col1, col2 = st.columns(2)
with col1:
    # Text area for matrix D input
    D_input = st.text_area("Matrix D (row-wise)", value=default_D)
with col2:
    # Text area for basis vectors input
    basis_input = st.text_area("Basis Vectors eᵢ (row-wise)", value=default_basis)

try:
    # Parse the matrix D input string into a numeric matrix
    D = parse_matrix_input(D_input, n)
    # Parse the basis vectors input string into a list of vectors
    basis = [parse_vector_input(row) for row in basis_input.strip().split(';')]
    # Compute symbolic Katz derivatives
    derivs = get_katz_derivatives(n, t_sym, a_val, D, basis)
    # Check cyclicity using the symbolic derivatives at t_val
    sym_result = check_cyclicity(derivs, t_sym, t_val)

    # Create two tabs: Symbolic Computation and ML Prediction
    tab1, tab2 = st.tabs([" Symbolic Computation", " ML Prediction"])
    with tab1:
        # Subheader for symbolic computation
        st.subheader(" Symbolic Katz Computation")
        # Show each derivative vector in an expander
        for i, dvec in enumerate(derivs):
            with st.expander(f"D^{i}(c)"):
                st.code(str(dvec))
        # Show rank and cyclicity result as metrics
        col1, col2 = st.columns(2)
        col1.metric("Rank at t", sym_result["rank"])
        col2.metric("Cyclic?", " Yes" if sym_result['is_cyclic'] else " No")
        # Show the evaluated derivative matrix as a dataframe
        with st.expander("🔎 Derivative Matrix at t"):
            mat_list = sym_result['matrix'].tolist()
            mat_numpy = np.array([[float(entry) for entry in row] for row in mat_list])
            mat_df = pd.DataFrame(mat_numpy).round(4)
            st.dataframe(mat_df)
    with tab2:
        # Subheader for ML prediction
        st.subheader("🤖 ML Model Prediction")
        # Flatten matrix D and basis vectors for ML input
        D_flat = flatten_matrix(D)
        B_flat = flatten_matrix(basis)

        # Define input feature column names
        input_columns = (
            [f'D_{i}{j}' for i in range(n) for j in range(n)] +
            [f'e{i}_{j}' for i in range(n) for j in range(n)] +
            ['a', 't']
        )
        # Prepare input values for the ML model
        input_values = [D_flat + B_flat + [a_val, t_val]]
        input_row = pd.DataFrame(input_values, columns=input_columns)
        # Add extra matrix features for the model
        input_row = add_matrix_features(input_row, n=n)
        # Ensure dataframe is Arrow-safe for Streamlit
        input_row = make_dataframe_arrow_safe(input_row)
        # Reorder columns if model expects a specific order
        if hasattr(model, 'feature_names_in_'):
            input_row = input_row[model.feature_names_in_]
        # Make prediction using the ML model
        prediction = model.predict(input_row)[0]
        # Show predicted class as a metric
        st.metric("Predicted Class", " Cyclic" if prediction else " Not Cyclic")
        # Show the ML input features in an expander
        with st.expander("📋 ML Input Features"):
            st.dataframe(input_row)
        # Show symbolic and ML results for comparison
        st.markdown(f"**Symbolic Result:** {' Cyclic' if sym_result['is_cyclic'] else ' Not Cyclic'}")
        st.markdown(f"**ML Prediction:** {' Cyclic' if prediction else ' Not Cyclic'}")
        # Warn if symbolic and ML results do not match
        if prediction != sym_result['is_cyclic']:
            st.warning(" Mismatch between Symbolic and ML prediction. Model may need retraining.")
except Exception as e:
    # Show error message if any exception occurs
    st.error(f"Error:{e}")