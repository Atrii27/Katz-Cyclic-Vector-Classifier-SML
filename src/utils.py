# === src/utils.py ===
import numpy as np
from sympy import Matrix

def parse_matrix_input(text, n):
    rows = text.strip().split(';')
    return Matrix([[eval(x) for x in row.split(',')] for row in rows])

def parse_vector_input(text):
    return Matrix([[eval(x.strip())] for x in text.split(',')])

def flatten_matrix(mat):
    if isinstance(mat, list):  # For basis = [Matrix, Matrix, ...]
        return [float(val[0]) if isinstance(val, Matrix) else float(val.evalf()) for vec in mat for val in vec]
    else:
        return [float(val[0]) if isinstance(val, Matrix) else float(val.evalf()) for val in mat]

def make_dataframe_arrow_safe(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x:
            float(x[0]) if isinstance(x, list) and len(x) == 1 else
            float(x) if hasattr(x, "__float__") else np.nan
        )
    return df.astype('float64')
