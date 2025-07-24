# === src/features.py ===
import pandas as pd
import numpy as np
from numpy.linalg import norm, eigvals, matrix_rank
def load_dataset(csv_path):
    # Load a CSV file into a pandas DataFrame
    return pd.read_csv(csv_path)
def get_feature_target_split(df):
    # Split DataFrame into features (X) and target (y)
    X = df.drop(columns=['cyclic'])
    y = df['cyclic']
    return X, y
def add_matrix_features(df, n=2):
    # Add matrix-based features to the DataFrame for ML
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.copy()  # Avoid modifying original DataFrame
    # Generate column names for D (matrix) and B (basis)
    D_cols = [f'D_{i}{j}' for i in range(n) for j in range(n)]
    B_cols = [f'e{i}_{j}' for i in range(n) for j in range(n)]
    # Ensure relevant columns are float type
    df[D_cols + B_cols + ['a', 't']] = df[D_cols + B_cols + ['a', 't']].astype(float)
    # Reshape flat columns into matrices for each row
    D_matrices = df[D_cols].values.reshape((-1, n, n))
    B_matrices = df[B_cols].values.reshape((-1, n, n))
    # Prepare lists to store computed features
    traces, dets, ranks = [], [], []
    eigval_1, eigval_2 = [], []
    basis_norms, basis_ranks = [], []
    # Compute features for each sample
    for D, B in zip(D_matrices, B_matrices):
        traces.append(np.trace(D))                 # Trace of D
        dets.append(np.linalg.det(D))              # Determinant of D
        ranks.append(matrix_rank(D))               # Rank of D
        eigs = np.real(eigvals(D))                 # Eigenvalues of D (real part)
        eigval_1.append(eigs[0] if len(eigs) > 0 else 0.0)  # First eigenvalue
        eigval_2.append(eigs[1] if len(eigs) > 1 else 0.0)  # Second eigenvalue
        norms = [norm(B[:, i]) for i in range(n)]           # Norms of basis vectors
        basis_norms.append(np.mean(norms))                  # Mean norm of basis
        basis_ranks.append(matrix_rank(B))                  # Rank of basis matrix
    # Add computed features to DataFrame
    df['trace_D'] = traces
    df['det_D'] = dets
    df['rank_D'] = ranks
    df['eigval_1'] = eigval_1
    df['eigval_2'] = eigval_2
    df['basis_norm_mean'] = basis_norms
    df['basis_rank'] = basis_ranks
    return df
def normalize_features(X):
    # Normalize features using StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)