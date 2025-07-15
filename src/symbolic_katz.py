# src/symbolic_katz.py

import sympy as sp
from sympy import Matrix, binomial, factorial, simplify

def compute_cij_table(n, D, basis):
    """
    Recursive computation of c(i, j) vectors for Katz's construction.
    """
    cij = {}

    # Base case: c(0, j)
    for j in range(n):
        vec = Matrix.zeros(n, 1)
        for k in range(j + 1):
            coeff = (-1)**k * binomial(j, k)
            ej_k = basis[j - k]
            term = ej_k
            for _ in range(k):
                term = D * term
            vec += coeff * term
        cij[(0, j)] = vec

    # Recursive step: c(i+1, j) = D(c(i, j)) + c(i, j+1)
    for i in range(1, n):
        for j in range(n):
            left = D * cij.get((i - 1, j), Matrix.zeros(n, 1))
            right = cij.get((i - 1, j + 1), Matrix.zeros(n, 1))
            cij[(i, j)] = left + right

    return cij

def compute_Di_c(i, t_sym, a_val, cij_table):
    """
    Compute D^i(c(t - a)) using the precomputed c(i, j) table.
    """
    vec = Matrix.zeros(cij_table[(0, 0)].rows, 1)
    x = t_sym - a_val
    for j in range(len([k for k in cij_table.keys() if k[0] == 0])):
        coeff = (x**j) / factorial(j)
        vec += coeff * cij_table[(i, j)]
    return simplify(vec)

def get_katz_derivatives(n, t_sym, a_val, D, basis):
    """
    Return [c, Dc, D^2c, ..., D^{n-1}c]
    """
    cij_table = compute_cij_table(n, D, basis)
    return [compute_Di_c(i, t_sym, a_val, cij_table) for i in range(n)]

def check_cyclicity(derivatives, t_sym, t_val):
    """
    Evaluate Katz vector matrix and compute rank at fixed t.
    """
    mat = Matrix.hstack(*derivatives)
    mat_eval = mat.subs(t_sym, t_val).evalf()
    rank = mat_eval.rank()
    is_cyclic = int(rank == mat_eval.shape[0])
    return {
        "rank": rank,
        "is_cyclic": is_cyclic,
        "matrix": mat_eval
    }
