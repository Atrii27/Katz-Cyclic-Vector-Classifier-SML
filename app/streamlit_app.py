# === app/streamlit_app.py ===

import streamlit as st
st.set_page_config(page_title="Katz Cyclic Vector Classifier (n=3)", layout="wide")

import joblib
import os
import pandas as pd
import numpy as np
from sympy import Matrix, symbols
import sys

# === Path Fix ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.symbolic_katz import get_katz_derivatives, check_cyclicity
from src.utils import parse_matrix_input, parse_vector_input, flatten_matrix, make_dataframe_arrow_safe
from src.features import add_matrix_features

# === Load ML model ===
model_path = os.path.join("models", "trained_model.pkl")
model = joblib.load(model_path)
st.sidebar.success(" Loaded ML model for n = 3")

# === App Title ===
st.markdown("""
    <h1 style='text-align: center; color: teal;'> Katz Cyclic Vector Classifier (n = 3)</h1>
    <p style='text-align: center;'>Compare symbolic cyclicity with ML prediction</p>
""", unsafe_allow_html=True)

# === Sidebar Inputs ===
n = 3
t_sym = symbols('t')
t_val = st.sidebar.slider("Evaluation point t", min_value=-5.0, max_value=5.0, value=1.0, step=0.5)
a_val = st.sidebar.slider("Shift value a", min_value=-5.0, max_value=5.0, value=1.0, step=0.5)

default_D = "1,1,0;0,1,1;0,0,1"
default_basis = "1,0,0;0,1,0;0,0,1"

col1, col2 = st.columns(2)
with col1:
    D_input = st.text_area("Matrix D (row-wise)", value=default_D)
with col2:
    basis_input = st.text_area("Basis Vectors eᵢ (row-wise)", value=default_basis)

try:
    # === Parse Inputs ===
    D = parse_matrix_input(D_input, n)
    basis = [parse_vector_input(row) for row in basis_input.strip().split(';')]

    # === Symbolic Katz Derivatives ===
    derivs = get_katz_derivatives(n, t_sym, a_val, D, basis)
    sym_result = check_cyclicity(derivs, t_sym, t_val)

    # === Tabs ===
    tab1, tab2 = st.tabs([" Symbolic Computation", " ML Prediction"])

    with tab1:
        st.subheader(" Symbolic Katz Computation")

        for i, dvec in enumerate(derivs):
            with st.expander(f"D^{i}(c)"):
                st.code(str(dvec))

        col1, col2 = st.columns(2)
        col1.metric("Rank at t", sym_result["rank"])
        col2.metric("Cyclic?", " Yes" if sym_result['is_cyclic'] else " No")

        with st.expander("🔎 Derivative Matrix at t"):
            mat_list = sym_result['matrix'].tolist()
            mat_numpy = np.array([[float(entry) for entry in row] for row in mat_list])
            mat_df = pd.DataFrame(mat_numpy).round(4)
            st.dataframe(mat_df)

    with tab2:
        st.subheader("🤖 ML Model Prediction")

        D_flat = flatten_matrix(D)
        B_flat = flatten_matrix(basis)

        input_columns = (
            [f'D_{i}{j}' for i in range(n) for j in range(n)] +
            [f'e{i}_{j}' for i in range(n) for j in range(n)] +
            ['a', 't']
        )
        input_values = [D_flat + B_flat + [a_val, t_val]]
        input_row = pd.DataFrame(input_values, columns=input_columns)

        input_row = add_matrix_features(input_row, n=n)
        input_row = make_dataframe_arrow_safe(input_row)

        if hasattr(model, 'feature_names_in_'):
            input_row = input_row[model.feature_names_in_]

        prediction = model.predict(input_row)[0]
        st.metric("Predicted Class", " Cyclic" if prediction else " Not Cyclic")

        with st.expander("📋 ML Input Features"):
            st.dataframe(input_row)

        # === Mismatch warning
        st.markdown(f"**Symbolic Result:** {' Cyclic' if sym_result['is_cyclic'] else ' Not Cyclic'}")
        st.markdown(f"**ML Prediction:** {' Cyclic' if prediction else ' Not Cyclic'}")

        if prediction != sym_result['is_cyclic']:
            st.warning(" Mismatch between Symbolic and ML prediction. Model may need retraining.")

except Exception as e:
    st.error(f" Error: {e}")
