# === src/features.py ===

import pandas as pd
import numpy as np
from numpy.linalg import norm, eigvals, matrix_rank

def load_dataset(csv_path):
    return pd.read_csv(csv_path)

def get_feature_target_split(df):
    X = df.drop(columns=['cyclic'])
    y = df['cyclic']
    return X, y

def add_matrix_features(df, n=2):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = df.copy()

    D_cols = [f'D_{i}{j}' for i in range(n) for j in range(n)]
    B_cols = [f'e{i}_{j}' for i in range(n) for j in range(n)]

    df[D_cols + B_cols + ['a', 't']] = df[D_cols + B_cols + ['a', 't']].astype(float)

    D_matrices = df[D_cols].values.reshape((-1, n, n))
    B_matrices = df[B_cols].values.reshape((-1, n, n))

    traces, dets, ranks = [], [], []
    eigval_1, eigval_2 = [], []
    basis_norms, basis_ranks = [], []

    for D, B in zip(D_matrices, B_matrices):
        traces.append(np.trace(D))
        dets.append(np.linalg.det(D))
        ranks.append(matrix_rank(D))

        eigs = np.real(eigvals(D))
        eigval_1.append(eigs[0] if len(eigs) > 0 else 0.0)
        eigval_2.append(eigs[1] if len(eigs) > 1 else 0.0)

        norms = [norm(B[:, i]) for i in range(n)]
        basis_norms.append(np.mean(norms))

        basis_ranks.append(matrix_rank(B))  # ✅ New feature

    df['trace_D'] = traces
    df['det_D'] = dets
    df['rank_D'] = ranks
    df['eigval_1'] = eigval_1
    df['eigval_2'] = eigval_2
    df['basis_norm_mean'] = basis_norms
    df['basis_rank'] = basis_ranks  # ✅ Added here

    return df

def normalize_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)
