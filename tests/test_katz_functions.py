# tests/test_katz_functions.py

import unittest
from sympy import Matrix, symbols
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.symbolic_katz import get_katz_derivatives, check_cyclicity

class TestKatzCyclicVectorN3(unittest.TestCase):
    def setUp(self):
        self.n = 3
        self.t_sym = symbols('t')
        self.t_val = 1.0

    def test_known_cyclic_example(self):
        D = Matrix([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1]
        ])
        basis = [
            Matrix([[1], [0], [0]]),
            Matrix([[0], [1], [0]]),
            Matrix([[0], [0], [1]])
        ]
        a_val = 1
        derivs = get_katz_derivatives(self.n, self.t_sym, a_val, D, basis)
        result = check_cyclicity(derivs, self.t_sym, self.t_val)
        self.assertEqual(result['is_cyclic'], 1)

    def test_known_noncyclic_example(self):
        D = Matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        basis = [
            Matrix([[1], [0], [0]]),
            Matrix([[0], [1], [0]]),
            Matrix([[0], [0], [0]])  # rank-deficient
        ]
        a_val = 1
        derivs = get_katz_derivatives(self.n, self.t_sym, a_val, D, basis)
        result = check_cyclicity(derivs, self.t_sym, self.t_val)
        self.assertEqual(result['is_cyclic'], 0)

if __name__ == '__main__':
    unittest.main()
