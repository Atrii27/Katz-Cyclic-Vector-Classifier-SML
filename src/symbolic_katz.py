# src/symbolic_katz.py
import sympy as sp
from sympy import Matrix, binomial, factorial, simplify
def compute_cij_table(n, D, basis):
    """
    Recursive computation of c(i, j) vectors for Katz's construction.
    """
    cij = {}  # Dictionary to store c(i, j) vectors
    # Base case: c(0, j)
    for j in range(n):
        vec = Matrix.zeros(n, 1)  # Initialize zero vector
        for k in range(j + 1):
            coeff = (-1)**k * binomial(j, k)  # Compute binomial coefficient with sign
            ej_k = basis[j - k]               # Select basis vector
            term = ej_k
            for _ in range(k):
                term = D * term               # Apply D k times
            vec += coeff * term               # Add term to vector
        cij[(0, j)] = vec                     # Store c(0, j)
    # Recursive step: c(i+1, j) = D(c(i, j)) + c(i, j+1)
    for i in range(1, n):
        for j in range(n):
            left = D * cij.get((i - 1, j), Matrix.zeros(n, 1))         # D applied to c(i-1, j)
            right = cij.get((i - 1, j + 1), Matrix.zeros(n, 1))        # c(i-1, j+1)
            cij[(i, j)] = left + right                                 # Sum for c(i, j)

    return cij  # Return the table of c(i, j) vectors
def compute_Di_c(i, t_sym, a_val, cij_table):
    """
    Compute D^i(c(t - a)) using the precomputed c(i, j) table.
    """
    vec = Matrix.zeros(cij_table[(0, 0)].rows, 1)  # Initialize zero vector of correct size
    x = t_sym - a_val                              # Compute (t - a)
    for j in range(len([k for k in cij_table.keys() if k[0] == 0])):  # Loop over j
        coeff = (x**j) / factorial(j)              # Taylor coefficient
        vec += coeff * cij_table[(i, j)]           # Add weighted c(i, j)
    return simplify(vec)                           # Simplify and return

def get_katz_derivatives(n, t_sym, a_val, D, basis):
    """
    Return [c, Dc, D^2c, ..., D^{n-1}c]
    """
    cij_table = compute_cij_table(n, D, basis)     # Precompute c(i, j) table
    return [compute_Di_c(i, t_sym, a_val, cij_table) for i in range(n)]  # Compute derivatives

def check_cyclicity(derivatives, t_sym, t_val):
    """
    Evaluate Katz vector matrix and compute rank at fixed t.
    """
    mat = Matrix.hstack(*derivatives)              # Stack derivatives as columns
    mat_eval = mat.subs(t_sym, t_val).evalf()      # Substitute t and evaluate numerically
    rank = mat_eval.rank()                         # Compute matrix rank
    is_cyclic = int(rank == mat_eval.shape[0])     # Check if full rank (cyclic)
    return {
        "rank": rank,                              # Matrix rank
        "is_cyclic": is_cyclic,                    # 1 if cyclic, 0 otherwise
        "matrix": mat_eval                         # Evaluated matrix
    }