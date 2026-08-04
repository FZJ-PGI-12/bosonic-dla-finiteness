import numpy as np
import pytest

from bosonic_dla_finiteness.algebra.free_hamiltonian import (
    FreeHamiltonian,
    compute_chi_freehamiltonian_gamma,
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

    def test_single_vector_is_wrapped(self):
        config = self._base_config()
        assert config.get_F() == [[1.0, 2.0]]

    def test_list_of_vectors_passed_through(self):
        config = self._base_config(omegas=[[1.0, 0.0], [0.0, 1.0]])
        assert config.get_F() == [[1.0, 0.0], [0.0, 1.0]]

    def test_validation_wrong_length(self):
        with pytest.raises(Exception):
            self._base_config(omegas=[1.0, 0.0, 0.0])  # length 3 ≠ n_modes=2

    def test_validation_wrong_length_in_list_form(self):
        with pytest.raises(Exception):
            # Two vectors is fine; the second having length 3 is not.
            self._base_config(omegas=[[1.0, 0.0], [0.0, 1.0, 0.0]])


class TestFreeHamiltonianDunders:
    def test_repr_roundtrips_coefficients(self):
        assert (
            repr(FreeHamiltonian([1.0, 2.5])) == "FreeHamiltonian([1.0, 2.5])"
        )

    def test_eq_returns_notimplemented_for_other_type(self):
        fh = FreeHamiltonian([1.0])
        assert fh.__eq__("not a free Hamiltonian") is NotImplemented
        assert fh != "not a free Hamiltonian"

    def test_hash_matches_for_equal_coefficients(self):
        assert hash(FreeHamiltonian([1.0, 2.0])) == hash(
            FreeHamiltonian([1.0, 2.0])
        )

    def test_usable_in_set(self):
        a = FreeHamiltonian([1.0, 2.0])
        b = FreeHamiltonian([1.0, 2.0])
        c = FreeHamiltonian([2.0, 1.0])
        assert len({a, b, c}) == 2


class TestChiGammaModeMismatch:
    """chi_X(gamma) is only defined when gamma has as many modes as X."""

    def test_gamma_longer_than_hamiltonian_rejected(self):
        fh = FreeHamiltonian([1.0, 2.0])
        with pytest.raises(ValueError, match="n=2 modes"):
            compute_chi_freehamiltonian_gamma(fh, ((1, 0, 0), (0, 0, 0)))

    def test_gamma_shorter_than_hamiltonian_rejected(self):
        fh = FreeHamiltonian([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="n=3 modes"):
            compute_chi_freehamiltonian_gamma(fh, ((1,), (0,)))

    def test_mismatched_beta_alone_rejected(self):
        fh = FreeHamiltonian([1.0, 2.0])
        with pytest.raises(ValueError, match="β of length 3"):
            compute_chi_freehamiltonian_gamma(fh, ((1, 0), (0, 0, 0)))
