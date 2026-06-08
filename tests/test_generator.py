import pytest

from bosonic_dla_finiteness.io.models import GeneratorKind
from bosonic_dla_finiteness.operators.monomial import tau
from bosonic_dla_finiteness.operators.operator import (
    BosonicGenerator,
    _is_canonical,
)


class TestIsCanonical:
    def test_diagonal_is_canonical(self):
        assert _is_canonical((1, 0), (1, 0))

    def test_zero_is_canonical(self):
        assert _is_canonical((0,), (0,))

    def test_canonical_hopping(self):
        # (ι_0, ι_1) in n=2: (1,0,0,1) > (0,1,1,0) → canonical
        assert _is_canonical((1, 0), (0, 1))

    def test_swapped_hopping_not_canonical(self):
        assert not _is_canonical((0, 1), (1, 0))


class TestGPlus:
    def test_number_operator(self):
        gamma = tau(0, 2)
        g = BosonicGenerator.g_plus(gamma)
        assert g.kind == GeneratorKind.plus
        assert g.gamma == gamma
        assert g.degree == 2
        assert g.n == 2

    def test_constant(self):
        g = BosonicGenerator.g_plus(((0,), (0,)))
        assert g.degree == 0

    def test_squeezing(self):
        # α=(2,0), β=(0,0): (2,0,0,0) > (0,0,2,0) → canonical
        g = BosonicGenerator.g_plus(((2, 0), (0, 0)))
        assert g.degree == 2

    def test_rejects_non_canonical(self):
        with pytest.raises(AssertionError):
            BosonicGenerator.g_plus(((0, 1), (1, 0)))


class TestGMinus:
    def test_hopping(self):
        gamma = ((1, 0), (0, 1))
        g = BosonicGenerator.g_minus(gamma)
        assert g.kind == GeneratorKind.minus
        assert g.degree == 2

    def test_rejects_diagonal(self):
        with pytest.raises(AssertionError):
            BosonicGenerator.g_minus(tau(0, 2))

    def test_rejects_non_canonical(self):
        with pytest.raises(AssertionError):
            BosonicGenerator.g_minus(((0, 1), (1, 0)))


class TestEqualityAndHash:
    def test_equal(self):
        g1 = BosonicGenerator.g_plus(tau(0, 2))
        g2 = BosonicGenerator.g_plus(tau(0, 2))
        assert g1 == g2
        assert hash(g1) == hash(g2)

    def test_different_kind(self):
        gamma = ((1, 0), (0, 1))
        assert BosonicGenerator.g_plus(gamma) != BosonicGenerator.g_minus(
            gamma
        )

    def test_different_gamma(self):
        assert BosonicGenerator.g_plus(tau(0, 2)) != BosonicGenerator.g_plus(
            tau(1, 2)
        )

    def test_usable_in_set(self):
        g1 = BosonicGenerator.g_plus(tau(0, 2))
        g2 = BosonicGenerator.g_plus(tau(0, 2))
        g3 = BosonicGenerator.g_plus(tau(1, 2))
        assert len({g1, g2, g3}) == 2
