import numpy as np
import pytest

from bosonic_dla_finiteness.algebra.free_hamiltonian import (
    FreeHamiltonian,
    compute_F_prime,
)
from bosonic_dla_finiteness.io.models import SystemConfig


class TestFreeHamiltonian:
    def test_from_omegas(self):
        fh = FreeHamiltonian.from_omegas([1.0, 2.0, 3.0])
        assert fh.n == 3
        assert list(fh.coeffs) == [1.0, 2.0, 3.0]

    def test_is_zero(self):
        assert FreeHamiltonian([0.0, 0.0]).is_zero()
        assert not FreeHamiltonian([1.0, 0.0]).is_zero()

    def test_equality(self):
        assert FreeHamiltonian([1.0, 2.0]) == FreeHamiltonian([1.0, 2.0])
        assert FreeHamiltonian([1.0, 0.0]) != FreeHamiltonian([0.0, 1.0])


class TestComputeFPrime:
    def test_single_nonzero(self):
        F = [FreeHamiltonian([1.0, 2.0, 3.0])]
        Fp = compute_F_prime(F)
        assert len(Fp) == 1

    def test_two_independent(self):
        F = [FreeHamiltonian([1.0, 0.0]), FreeHamiltonian([0.0, 1.0])]
        Fp = compute_F_prime(F)
        assert len(Fp) == 2

    def test_two_dependent(self):
        # Second is 2× the first
        F = [FreeHamiltonian([1.0, 2.0]), FreeHamiltonian([2.0, 4.0])]
        Fp = compute_F_prime(F)
        assert len(Fp) == 1

    def test_span_preserved(self):
        x1 = np.array([1.0, 1.0, 0.0])
        x2 = np.array([1.0, 0.0, 1.0])
        F = [FreeHamiltonian(x1), FreeHamiltonian(x2)]
        Fp = compute_F_prime(F)
        assert len(Fp) == 2
        # Span of F' must contain x1 and x2: check residual of least-squares fit
        basis = np.array([fh.coeffs for fh in Fp])
        for x in [x1, x2]:
            coeffs, _, _, _ = np.linalg.lstsq(basis.T, x, rcond=None)
            assert np.linalg.norm(basis.T @ coeffs - x) < 1e-10

    def test_empty_F(self):
        assert compute_F_prime([]) == []

    def test_all_zero(self):
        F = [FreeHamiltonian([0.0, 0.0]), FreeHamiltonian([0.0, 0.0])]
        assert compute_F_prime(F) == []

    def test_three_vectors_rank_two(self):
        F = [
            FreeHamiltonian([1.0, 0.0, 0.0]),
            FreeHamiltonian([0.0, 1.0, 0.0]),
            FreeHamiltonian(
                [1.0, 1.0, 0.0]
            ),  # linear combination of the first two
        ]
        Fp = compute_F_prime(F)
        assert len(Fp) == 2


class TestSystemConfigGetF:
    def _base_config(self, **kwargs):
        defaults = {
            "n_modes": 2,
            "omegas": [1.0, 2.0],
            "generators": [
                {"kind": "+", "alpha": [1, 0], "beta": [0, 1], "label": "G1"}
            ],
        }
        defaults.update(kwargs)
        return SystemConfig(**defaults)

    def test_default_F_is_omegas(self):
        config = self._base_config()
        assert config.get_F() == [[1.0, 2.0]]

    def test_explicit_F(self):
        config = self._base_config(free_hamiltonians=[[1.0, 0.0], [0.0, 1.0]])
        assert config.get_F() == [[1.0, 0.0], [0.0, 1.0]]

    def test_validation_wrong_length(self):
        with pytest.raises(Exception):
            self._base_config(
                free_hamiltonians=[[1.0, 0.0, 0.0]]
            )  # length 3 ≠ n_modes=2
