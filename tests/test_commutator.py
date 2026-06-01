from bosonic_dla_finiteness.io.models import GeneratorKind
from bosonic_dla_finiteness.operators.commutator import (
    commutator_coeffs,
    commutator_support,
    normal_order_product,
)
from bosonic_dla_finiteness.operators.monomial import tau, zero_gamma
from bosonic_dla_finiteness.operators.operator import BosonicGenerator

_TOL = 1e-10


class TestNormalOrderProduct:
    def test_identity_left(self):
        # 1 · a^γ = a^γ
        gamma = ((1,), (0,))
        result = normal_order_product(zero_gamma(1), gamma)
        assert abs(result.get(gamma, 0j) - 1.0) < _TOL
        assert len(result) == 1

    def test_identity_right(self):
        gamma = ((1,), (0,))
        result = normal_order_product(gamma, zero_gamma(1))
        assert abs(result.get(gamma, 0j) - 1.0) < _TOL

    def test_creators_add(self):
        # a†· a† = (a†)^2  — no contractions (both creators)
        cre = ((1,), (0,))
        result = normal_order_product(cre, cre)
        assert abs(result.get(((2,), (0,)), 0j) - 1.0) < _TOL

    def test_CCR_annihilator_then_creator(self):
        # a · a† = a†a + 1  →  {((1,),(1,)): 1, ((0,),(0,)): 1}
        ann = ((0,), (1,))
        cre = ((1,), (0,))
        result = normal_order_product(ann, cre)
        assert abs(result.get(((1,), (1,)), 0j) - 1.0) < _TOL
        assert abs(result.get(((0,), (0,)), 0j) - 1.0) < _TOL

    def test_creator_then_annihilator_no_constant(self):
        # a† · a already in normal order → a†a, no constant term
        cre = ((1,), (0,))
        ann = ((0,), (1,))
        result = normal_order_product(cre, ann)
        assert abs(result.get(((1,), (1,)), 0j) - 1.0) < _TOL
        assert ((0,), (0,)) not in result or abs(result[((0,), (0,))]) < _TOL

    def test_multimode_independent(self):
        # a_0 · a†_1 = a†_1 · a_0 (different modes commute — no CCR contraction)
        ann0 = ((0, 0), (1, 0))
        cre1 = ((0, 1), (0, 0))
        result = normal_order_product(ann0, cre1)
        assert abs(result.get(((0, 1), (1, 0)), 0j) - 1.0) < _TOL
        assert len(result) == 1


class TestCommutatorCoeffs:
    def test_self_commutator_zero(self):
        g = BosonicGenerator.g_plus(tau(0, 1))
        assert all(abs(v) < _TOL for v in commutator_coeffs(g, g).values())

    def test_antisymmetry(self):
        g1 = BosonicGenerator.g_plus(tau(0, 2))
        g2 = BosonicGenerator.g_minus(((1, 0), (0, 1)))
        c12 = commutator_coeffs(g1, g2)
        c21 = commutator_coeffs(g2, g1)
        all_keys = set(c12) | set(c21)
        for k in all_keys:
            assert abs(c12.get(k, 0.0) + c21.get(k, 0.0)) < _TOL

    def test_free_with_linear_generator(self):
        # [g_+^{τ_0}, g_-^{ι_0}] = -2 g_+^{ι_0}  (1-mode)
        #
        # g_+^{τ_0} = 2i a†a,   g_-^{ι_0} = a - a†
        # [2i a†a, a − a†] = 2i(-a − a†) = -2 · i(a + a†) = -2 g_+^{ι_0}
        n = 1
        g_tau = BosonicGenerator.g_plus(tau(0, n))
        g_lin = BosonicGenerator.g_minus(((1,), (0,)))
        coeffs = commutator_coeffs(g_tau, g_lin)
        key = (GeneratorKind.plus, ((1,), (0,)))
        assert abs(coeffs.get(key, 0.0) - (-2.0)) < _TOL
        assert len(coeffs) == 1


class TestCommutatorSupport:
    def test_support_matches_nonzero_coeffs(self):
        g1 = BosonicGenerator.g_plus(tau(0, 1))
        g2 = BosonicGenerator.g_minus(((1,), (0,)))
        coeffs = commutator_coeffs(g1, g2)
        support = commutator_support(g1, g2)
        assert {(g.kind, g.gamma) for g in support} == set(coeffs.keys())

    def test_zero_commutator_empty_support(self):
        g = BosonicGenerator.g_plus(tau(0, 1))
        assert commutator_support(g, g) == frozenset()
