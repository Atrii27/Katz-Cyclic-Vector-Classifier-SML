# === scripts/generate_dataset.py ===

import sympy as sp
import random
import csv
import os
from sympy import Matrix, symbols, binomial, factorial, simplify

def compute_cij_table(n, D, basis):
    cij = {}
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

    for i in range(1, n):
        for j in range(n):
            left = D * cij.get((i - 1, j), Matrix.zeros(n, 1))
            right = cij.get((i - 1, j + 1), Matrix.zeros(n, 1))
            cij[(i, j)] = left + right

    return cij

def compute_Di_c(i, t_sym, a_val, cij_table):
    vec = Matrix.zeros(cij_table[(0,0)].rows, 1)
    x = t_sym - a_val
    for j in range(len([k for k in cij_table.keys() if k[0] == 0])):
        coeff = (x**j) / factorial(j)
        vec += coeff * cij_table[(i, j)]
    return simplify(vec)

def get_katz_derivatives(n, t_sym, a_val, D, basis):
    cij_table = compute_cij_table(n, D, basis)
    derivatives = [compute_Di_c(i, t_sym, a_val, cij_table) for i in range(n)]
    return derivatives

def check_cyclicity(derivatives, t_sym, t_val):
    mat = Matrix.hstack(*derivatives)
    mat_eval = mat.subs(t_sym, t_val).evalf()
    rank = mat_eval.rank()
    return int(rank == mat_eval.shape[0])

def flatten_matrix(mat):
    return [float(val.evalf()) for row in mat.tolist() for val in row]

def flatten_basis(basis):
    return [float(val.evalf()) for vec in basis for val in vec]

def generate_random_D(n=3):
    return Matrix([[random.randint(0, 4) for _ in range(n)] for _ in range(n)])

def generate_random_basis(n=3):
    if random.random() < 0.2:
        # 20% chance of rank-deficient basis
        basis = [Matrix([[1 if i == j else 0] for i in range(n)]) for j in range(n)]
        basis[1] = Matrix.zeros(n, 1)
        return basis
    else:
        return [Matrix([[random.randint(0, 1)] for _ in range(n)]) for _ in range(n)]

def generate_dataset(output_file='katz_cyclic_vector_ml/data/raw/samples_n3.csv', num_samples=500, desired_ratio=0.5):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    t_sym = symbols('t')
    t_val = 1.0
    n = 3

    samples = []
    count_cyclic = 0
    count_noncyclic = 0
    target_cyclic = int(num_samples * desired_ratio)
    target_noncyclic = num_samples - target_cyclic

    attempts = 0
    max_attempts = 1500

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            D = generate_random_D(n)
            basis = generate_random_basis(n)
            a = random.randint(1, 5)

            derivs = get_katz_derivatives(n, t_sym, a, D, basis)
            label = check_cyclicity(derivs, t_sym, t_val)

            if (label == 1 and count_cyclic >= target_cyclic) or (label == 0 and count_noncyclic >= target_noncyclic):
                continue

            row = flatten_matrix(D) + flatten_basis(basis) + [a, t_val, label]
            samples.append(row)
            if label == 1:
                count_cyclic += 1
            else:
                count_noncyclic += 1

            print(f"✅ Sample {len(samples)} | Label={label} | Cyclic={count_cyclic} Non-Cyclic={count_noncyclic}")

        except Exception as e:
            print(f"⚠️ Skipped due to error: {e}")
            continue

    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'D_{i}{j}' for i in range(n) for j in range(n)] + \
                 [f'e{i}_{j}' for i in range(n) for j in range(n)] + \
                 ['a', 't', 'cyclic']
        writer.writerow(header)
        writer.writerows(samples)

    print(f"\n🎯 Dataset saved to: {output_file}")
    print(f"📊 Total: {len(samples)} | Cyclic: {count_cyclic}, Non-Cyclic: {count_noncyclic}")

if __name__ == "__main__":
    generate_dataset()
