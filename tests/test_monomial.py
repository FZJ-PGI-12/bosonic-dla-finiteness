"""
Tests for the bosonic_dla_finiteness.operators.monomial module.
"""

import pytest

from bosonic_dla_finiteness.operators.monomial import (
    gamma_degree,
    gamma_from_iotas_sum,
    iota,
    tau,
)

# ── Tests for iota ────────────────────────────────────────────────────────────


class TestIota:
    """Tests for iota function."""

    def test_iota_basic(self):
        """Test basic iota construction."""
        assert iota(0, 3) == (1, 0, 0)
        assert iota(1, 3) == (0, 1, 0)
        assert iota(2, 3) == (0, 0, 1)

    def test_iota_single_element(self):
        """Test iota with n=1."""
        assert iota(0, 1) == (1,)

    def test_iota_large_n(self):
        """Test iota with larger dimension."""
        result = iota(3, 5)
        assert result == (0, 0, 0, 1, 0)
        assert len(result) == 5

    def test_iota_all_zeros_except_one(self):
        """Test that iota has exactly one non-zero entry."""
        for k in range(5):
            result = iota(k, 5)
            assert sum(result) == 1
            assert result[k] == 1


# ── Tests for tau ────────────────────────────────────────────────────────────


class TestTau:
    """Tests for tau function."""

    def test_tau_basic(self):
        """Test basic tau construction."""
        result = tau(1, 3)
        assert result == ((0, 1, 0), (0, 1, 0))

    def test_tau_is_symmetric(self):
        """Test that tau_p has alpha == beta."""
        for p in range(5):
            alpha, beta = tau(p, 5)
            assert alpha == beta

    def test_tau_equals_iota_pair(self):
        """Test that tau_p = (iota_p, iota_p)."""
        for p in range(4):
            alpha, beta = tau(p, 4)
            assert alpha == iota(p, 4)
            assert beta == iota(p, 4)


# ── Tests for gamma_degree ────────────────────────────────────────────────────


class TestGammaDegree:
    """Tests for gamma_degree function."""

    def test_gamma_degree_basic(self):
        """Test basic degree computation."""
        gamma = ((1, 0, 0), (0, 1, 1))
        assert gamma_degree(gamma) == 3

    def test_gamma_degree_zero(self):
        """Test degree of zero monomial."""
        gamma = ((0, 0, 0), (0, 0, 0))
        assert gamma_degree(gamma) == 0

    def test_gamma_degree_identity(self):
        """Test degree of identity monomial."""
        assert gamma_degree(((0, 0, 0, 0, 0), (0, 0, 0, 0, 0))) == 0

    def test_gamma_degree_tau(self):
        """Test degree of tau monomial."""
        gamma = tau(2, 5)
        assert gamma_degree(gamma) == 2


class TestGammaFromIotasSumValidation:
    """
    Exponent lists must match their index lists. Raised, not asserted, so the
    check is not stripped by `python -O`.
    """

    def test_alpha_exponent_length_mismatch(self):
        with pytest.raises(ValueError, match="alpha_indices"):
            gamma_from_iotas_sum(n=3, alpha_indices=[0, 1], alpha_exps=[2])

    def test_beta_exponent_length_mismatch(self):
        with pytest.raises(ValueError, match="beta_indices"):
            gamma_from_iotas_sum(n=3, beta_indices=[0], beta_exps=[1, 1])
