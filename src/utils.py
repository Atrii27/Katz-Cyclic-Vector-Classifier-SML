# === src/utils.py ===
import numpy as np
from sympy import Matrix
# Parse a semicolon-separated string into a sympy Matrix of size n x n
def parse_matrix_input(text, n):
    rows = text.strip().split(';')  # Split input into rows
    return Matrix([[eval(x) for x in row.split(',')] for row in rows])  # Convert each entry to number
# Parse a comma-separated string into a sympy column vector
def parse_vector_input(text):
    return Matrix([[eval(x.strip())] for x in text.split(',')])  # Each value as a row in a column vector
# Flatten a matrix or list of vectors into a 1D list of floats
def flatten_matrix(mat):
    if isinstance(mat, list):  # If input is a list of vectors (e.g., basis)
        return [float(val[0]) if isinstance(val, Matrix) else float(val.evalf()) for vec in mat for val in vec]
    else:  # If input is a single matrix
        return [float(val[0]) if isinstance(val, Matrix) else float(val.evalf()) for val in mat]
# Convert all columns in a DataFrame to float64, handling lists and float-like objects
def make_dataframe_arrow_safe(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x:
            float(x[0]) if isinstance(x, list) and len(x) == 1 else  # Convert single-item lists to float
            float(x) if hasattr(x, "__float__") else np.nan          # Convert float-like objects, else NaN
        )
    return df.astype('float64')  # Ensure all