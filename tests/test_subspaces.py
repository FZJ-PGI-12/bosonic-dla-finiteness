from bosonic_dla_finiteness.algebra.subspaces import (
    Subspace,
    decompose_generators,
    determine_subspace,
)
from bosonic_dla_finiteness.operators.monomial import tau, zero_gamma
from bosonic_dla_finiteness.operators.operator import BosonicGenerator


def gp(gamma):
    return BosonicGenerator.g_plus(gamma)


def gm(gamma):
    return BosonicGenerator.g_minus(gamma)


class TestClassify:
    def test_G0_constant(self):
        # degree 0: constant 2i = g_+^{0}
        assert determine_subspace(gp(zero_gamma(2))) == Subspace.G0

    def test_G0_number_operator_mode0(self):
        assert determine_subspace(gp(tau(0, 2))) == Subspace.G0

    def test_G0_number_operator_mode1(self):
        assert determine_subspace(gp(tau(1, 2))) == Subspace.G0

    def test_G1_creation(self):
        # g_+^{(ι_0, 0)}: degree 1
        assert determine_subspace(gp(((1, 0), (0, 0)))) == Subspace.G1

    def test_G1_g_minus(self):
        assert determine_subspace(gm(((1, 0), (0, 0)))) == Subspace.G1

    def test_G2_hopping(self):
        # g_+^{(ι_0, ι_1)}: degree 2, α≠β
        assert determine_subspace(gp(((1, 0), (0, 1)))) == Subspace.G2

    def test_G2_g_minus_hopping(self):
        assert determine_subspace(gm(((1, 0), (0, 1)))) == Subspace.G2

    def test_G2_squeezing(self):
        # g_+^{((2,0),(0,0))}: degree 2, α≠β
        assert determine_subspace(gp(((2, 0), (0, 0)))) == Subspace.G2

    def test_Geq(self):
        # g_+^{((2,0),(2,0))}: degree 4, α=β
        assert determine_subspace(gp(((2, 0), (2, 0)))) == Subspace.Geq

    def test_Geq_single_mode_degree4(self):
        # g_+^{((2,),(2,))}: degree 4, n=1
        assert determine_subspace(gp(((2,), (2,)))) == Subspace.Geq

    def test_Gom(self):
        # n=2: α=(1,1), β=(0,1) → degree=3
        # mode 0: α_0+β_0 = 1+0 = 1 (unique offdiag)
        # mode 1: α_1=β_1=1 (diagonal)
        # canonical check: (1,1,0,1) vs (0,1,1,1) → (1,...) > (0,...) ✓
        assert determine_subspace(gp(((1, 1), (0, 1)))) == Subspace.Gom

    def test_Gperp(self):
        # n=2: α=(2,1), β=(1,0) → degree=4, not diagonal, two offdiag modes
        # canonical: (2,1,1,0) vs (1,0,2,1): first pos 2>1 ✓
        assert determine_subspace(gp(((2, 1), (1, 0)))) == Subspace.Gperp


class TestDecomposeGenerators:
    def _all_generators(self):
        return [
            gp(zero_gamma(2)),  # G0
            gp(tau(0, 2)),  # G0
            gp(((1, 0), (0, 0))),  # G1
            gm(((1, 0), (0, 0))),  # G1
            gp(((1, 0), (0, 1))),  # G2
            gm(((1, 0), (0, 1))),  # G2
            gp(((2, 0), (2, 0))),  # G=
            gp(((1, 1), (0, 1))),  # Gom
            gp(((2, 1), (1, 0))),  # Gperp
        ]

    def test_partition_covers_all(self):
        generators = self._all_generators()
        result = decompose_generators(generators)
        flat = [g for gs in result.values() for g in gs]
        assert len(flat) == len(generators)

    def test_partition_is_disjoint(self):
        generators = self._all_generators()
        result = decompose_generators(generators)
        all_assigned = [g for gs in result.values() for g in gs]
        assert len(all_assigned) == len(set(all_assigned))

    def test_correct_subspace_assignment(self):
        g_a0 = gp(tau(0, 2))
        g_a1 = gp(((1, 0), (0, 0)))
        g_a2 = gp(((1, 0), (0, 1)))
        result = decompose_generators([g_a0, g_a1, g_a2])
        assert g_a0 in result[Subspace.G0]
        assert g_a1 in result[Subspace.G1]
        assert g_a2 in result[Subspace.G2]

    def test_empty_input(self):
        result = decompose_generators([])
        assert all(len(v) == 0 for v in result.values())
