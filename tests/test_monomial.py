"""
Tests for the bosonic_dla_finiteness.operators.monomial module.
"""

import pytest

from bosonic_dla_finiteness.operators.monomial import (
    BosonicMonomial,
    gamma_add,
    gamma_adjoint,
    gamma_degree,
    gamma_modes,
    iota,
    is_self_adjoint_gamma,
    tau,
    zero_gamma,
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


# ── Tests for zero_gamma ──────────────────────────────────────────────────────


class TestZeroGamma:
    """Tests for zero_gamma function."""

    def test_zero_gamma_basic(self):
        """Test zero_gamma construction."""
        assert zero_gamma(3) == ((0, 0, 0), (0, 0, 0))

    def test_zero_gamma_single(self):
        """Test zero_gamma with n=1."""
        assert zero_gamma(1) == ((0,), (0,))

    def test_zero_gamma_length(self):
        """Test that zero_gamma has correct dimensions."""
        alpha, beta = zero_gamma(5)
        assert len(alpha) == 5
        assert len(beta) == 5


# ── Tests for BosonicMonomial ─────────────────────────────────────────────────


class TestBosonicMonomialInit:
    """Tests for BosonicMonomial initialization."""

    def test_init_valid_gamma(self):
        """Test initialization with valid gamma."""
        gamma = ((1, 0), (0, 1))
        m = BosonicMonomial(gamma)
        assert m.alpha == (1, 0)
        assert m.beta == (0, 1)

    def test_init_zero_gamma(self):
        """Test initialization with zero gamma (identity monomial)."""
        gamma = ((0, 0), (0, 0))
        m = BosonicMonomial(gamma)
        assert m.alpha == (0, 0)
        assert m.beta == (0, 0)

    def test_init_mismatched_lengths(self):
        """Test that mismatched alpha/beta lengths raise error."""
        with pytest.raises(AssertionError):
            BosonicMonomial(((1, 0), (0, 1, 0)))

    def test_init_negative_alpha(self):
        """Test that negative alpha entries raise error."""
        with pytest.raises(AssertionError):
            BosonicMonomial(((-1, 0), (0, 1)))

    def test_init_negative_beta(self):
        """Test that negative beta entries raise error."""
        with pytest.raises(AssertionError):
            BosonicMonomial(((1, 0), (-1, 1)))

    def test_init_converts_to_tuples(self):
        """Test that alpha and beta are stored as tuples."""
        gamma = ([1, 0], [0, 1])
        m = BosonicMonomial(gamma)
        assert isinstance(m.alpha, tuple)
        assert isinstance(m.beta, tuple)


class TestBosonicMonomialProperties:
    """Tests for BosonicMonomial properties."""

    def test_gamma_property(self):
        """Test that gamma is correctly computed."""
        m = BosonicMonomial(((1, 0), (0, 1)))
        assert m.gamma == ((1, 0), (0, 1))

    def test_gamma_property_zero(self):
        """Test gamma property for zero monomial."""
        m = BosonicMonomial(((0, 0), (0, 0)))
        assert m.gamma == ((0, 0), (0, 0))

    def test_gamma_property_large(self):
        """Test gamma property with larger dimensions."""
        m = BosonicMonomial(((2, 1, 0), (0, 1, 2)))
        assert m.gamma == ((2, 1, 0), (0, 1, 2))


class TestBosonicMonomialEquality:
    """Tests for BosonicMonomial equality and hashing."""

    def test_equal_monomials(self):
        """Test that equal monomials are equal."""
        m1 = BosonicMonomial(((1, 0), (0, 1)))
        m2 = BosonicMonomial(((1, 0), (0, 1)))
        assert m1 == m2

    def test_unequal_alpha(self):
        """Test that monomials with different alpha are not equal."""
        m1 = BosonicMonomial(((1, 0), (0, 1)))
        m2 = BosonicMonomial(((0, 1), (0, 1)))
        assert m1 != m2

    def test_unequal_beta(self):
        """Test that monomials with different beta are not equal."""
        m1 = BosonicMonomial(((1, 0), (0, 1)))
        m2 = BosonicMonomial(((1, 0), (1, 0)))
        assert m1 != m2

    def test_hashable(self):
        """Test that monomials are hashable."""
        m = BosonicMonomial(((1, 0), (0, 1)))
        assert hash(m) is not None

    def test_hash_equal_for_equal_monomials(self):
        """Test that equal monomials have equal hashes."""
        m1 = BosonicMonomial(((1, 0), (0, 1)))
        m2 = BosonicMonomial(((1, 0), (0, 1)))
        assert hash(m1) == hash(m2)

    def test_can_use_as_dict_key(self):
        """Test that monomials can be used as dictionary keys."""
        m1 = BosonicMonomial(((1, 0), (0, 1)))
        m2 = BosonicMonomial(((1, 0), (0, 1)))
        d = {m1: "value"}
        assert d[m2] == "value"


class TestBosonicMonomialRepr:
    """Tests for BosonicMonomial string representation."""

    def test_repr_format(self):
        """Test string representation format."""
        m = BosonicMonomial(((1, 0), (0, 1)))
        assert repr(m) == "a^((1, 0),(0, 1))"

    def test_repr_zero(self):
        """Test repr of zero monomial."""
        m = BosonicMonomial(((0, 0), (0, 0)))
        assert repr(m) == "a^((0, 0),(0, 0))"


class TestBosonicMonomialDagger:
    """Tests for BosonicMonomial.dagger() method."""

    def test_dagger_basic(self):
        """Test dagger operation."""
        m = BosonicMonomial(((1, 0), (0, 1)))
        m_dag = m.dagger()
        assert m_dag.alpha == (0, 1)
        assert m_dag.beta == (1, 0)

    def test_dagger_involution(self):
        """Test that dagger squared is identity."""
        m = BosonicMonomial(((1, 2), (3, 4)))
        assert m.dagger().dagger() == m

    def test_dagger_self_adjoint(self):
        """Test dagger of self-adjoint monomial."""
        m = BosonicMonomial(((1, 0), (1, 0)))
        m_dag = m.dagger()
        assert m_dag == m

    def test_dagger_returns_monomial(self):
        """Test that dagger returns a BosonicMonomial."""
        m = BosonicMonomial(((1, 0), (0, 1)))
        m_dag = m.dagger()
        assert isinstance(m_dag, BosonicMonomial)


# ── Tests for gamma functions ─────────────────────────────────────────────────


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
        assert gamma_degree(zero_gamma(5)) == 0

    def test_gamma_degree_tau(self):
        """Test degree of tau monomial."""
        gamma = tau(2, 5)
        assert gamma_degree(gamma) == 2


class TestGammaModes:
    """Tests for gamma_modes function."""

    def test_gamma_modes_basic(self):
        """Test basic mode computation."""
        gamma = ((1, 0, 0), (0, 1, 1))
        assert gamma_modes(gamma) == frozenset({0, 1, 2})

    def test_gamma_modes_zero(self):
        """Test modes of zero monomial."""
        gamma = zero_gamma(3)
        assert gamma_modes(gamma) == frozenset()

    def test_gamma_modes_single_mode(self):
        """Test modes with single non-zero entry."""
        gamma = ((1, 0, 0), (0, 0, 0))
        assert gamma_modes(gamma) == frozenset({0})

    def test_gamma_modes_tau(self):
        """Test modes of tau monomial."""
        gamma = tau(1, 5)
        assert gamma_modes(gamma) == frozenset({1})


class TestGammaAdjoint:
    """Tests for gamma_adjoint function."""

    def test_gamma_adjoint_basic(self):
        """Test basic adjoint computation."""
        gamma = ((1, 0, 0), (0, 1, 1))
        gamma_dag = gamma_adjoint(gamma)
        assert gamma_dag == ((0, 1, 1), (1, 0, 0))

    def test_gamma_adjoint_involution(self):
        """Test that adjoint squared is identity."""
        gamma = ((1, 2), (3, 4))
        assert gamma_adjoint(gamma_adjoint(gamma)) == gamma

    def test_gamma_adjoint_self_adjoint(self):
        """Test adjoint of self-adjoint gamma."""
        gamma = ((1, 0), (1, 0))
        assert gamma_adjoint(gamma) == gamma

    def test_gamma_adjoint_zero(self):
        """Test adjoint of zero gamma."""
        gamma = zero_gamma(3)
        assert gamma_adjoint(gamma) == gamma


class TestGammaAdd:
    """Tests for gamma_add function."""

    def test_gamma_add_basic(self):
        """Test basic gamma addition."""
        g1 = ((1, 0), (0, 1))
        g2 = ((0, 1), (1, 0))
        result = gamma_add(g1, g2)
        assert result == ((1, 1), (1, 1))

    def test_gamma_add_zero(self):
        """Test adding to zero gamma."""
        g = ((1, 0), (0, 1))
        z = zero_gamma(2)
        assert gamma_add(g, z) == g
        assert gamma_add(z, g) == g

    def test_gamma_add_associative(self):
        """Test that gamma addition is associative."""
        g1 = ((1, 0), (0, 1))
        g2 = ((0, 1), (1, 0))
        g3 = ((1, 1), (0, 0))
        assert gamma_add(gamma_add(g1, g2), g3) == gamma_add(
            g1, gamma_add(g2, g3)
        )

    def test_gamma_add_commutative(self):
        """Test that gamma addition is commutative."""
        g1 = ((1, 0), (0, 1))
        g2 = ((0, 1), (1, 0))
        assert gamma_add(g1, g2) == gamma_add(g2, g1)


class TestIsSelfAdjointGamma:
    """Tests for is_self_adjoint_gamma function."""

    def test_is_self_adjoint_true(self):
        """Test self-adjoint gamma."""
        gamma = ((1, 0), (1, 0))
        assert is_self_adjoint_gamma(gamma)

    def test_is_self_adjoint_false(self):
        """Test non-self-adjoint gamma."""
        gamma = ((1, 0), (0, 1))
        assert not is_self_adjoint_gamma(gamma)

    def test_is_self_adjoint_zero(self):
        """Test that zero gamma is self-adjoint."""
        gamma = zero_gamma(3)
        assert is_self_adjoint_gamma(gamma)

    def test_is_self_adjoint_tau(self):
        """Test that tau is self-adjoint."""
        gamma = tau(1, 5)
        assert is_self_adjoint_gamma(gamma)
