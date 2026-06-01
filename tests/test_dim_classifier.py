from bosonic_dla_finiteness.algebra.dim_classifier import (
    DimensionResult,
    mode_match,
)
from bosonic_dla_finiteness.algebra.free_hamiltonian import FreeHamiltonian
from bosonic_dla_finiteness.operators.monomial import (
    gamma_from_iotas,
    gamma_from_iotas_sum,
)
from bosonic_dla_finiteness.operators.operator import BosonicGenerator


class TestDimClassifier:
    def test_mode_match(self):
        fh = FreeHamiltonian([i for i in range(1, 13)])
        # g1: g_+(a†_0)         = i(a†_0 + a_0)                             — G1, linear on mode 0
        g1 = BosonicGenerator(
            kind="+", gamma=gamma_from_iotas(n=12, alpha_idx=0)
        )
        assert g1

        # g2: g_-(a†_11)        = a†_11 - a_11                              — G1, linear on mode 11
        g2 = BosonicGenerator(
            kind="-", gamma=gamma_from_iotas(n=12, alpha_idx=11)
        )
        assert g2

        # g3: g_+(a†_11)        = i(a†_11 + a_11)                           — G1, linear on mode 11
        g3 = BosonicGenerator(
            kind="+", gamma=gamma_from_iotas(n=12, alpha_idx=11)
        )
        assert g3

        # g4: g_-((a†_8)^2)     = (a†_8)^2 - a_8^2                         — G2, squeezing on mode 8
        g4 = BosonicGenerator(
            kind="-", gamma=gamma_from_iotas(n=12, alpha_idx=8, alpha_exp=2)
        )
        assert g4

        # g5: g_+((a†_8)^2)     = i((a†_8)^2 + a_8^2)                      — G2, squeezing on mode 8
        g5 = BosonicGenerator(
            kind="+", gamma=gamma_from_iotas(n=12, alpha_idx=8, alpha_exp=2)
        )
        assert g5

        # g6: g_-(a†_9 a†_11)   = a†_9 a†_11 - a_9 a_11                 — G2, two-mode squeezing modes 9,11
        g6 = BosonicGenerator(
            kind="-", gamma=gamma_from_iotas_sum(n=12, alpha_indices=[9, 11])
        )
        assert g6

        # g7: g_-(a†_3 a†_8 a_3) = a†_3 a_3 a_8 - a†_3 a†_8 a_3             — Gom, off-diagonal on mode 8 only
        g7 = BosonicGenerator(
            kind="-",
            gamma=gamma_from_iotas_sum(
                n=12, alpha_indices=[3, 8], beta_indices=[3]
            ),
        )
        assert g7

        # g8: g_+((a†_3)^2 a†_4 a†_10 (a_3)^2 a_4) = i((a†_3)^2 a†_4 (a_3)^2 a_4 a_10 + (a†_3)^2 a†_4 a†_10 (a_3)^2 a_4)  — Gom, degree 7
        g8 = BosonicGenerator(
            kind="+",
            gamma=gamma_from_iotas_sum(
                n=12,
                alpha_indices=[3, 4, 10],
                beta_indices=[3, 4],
                alpha_exps=[2, 1, 1],
                beta_exps=[2, 1],
            ),
        )
        assert g8

        # g9: g_+((a†_2)^2 (a_2)^2) = 2i(a†_2)^2(a_2)^2                   — Geq, diagonal degree 4 on mode 2
        g9 = BosonicGenerator(
            kind="+",
            gamma=gamma_from_iotas(
                n=12, alpha_idx=2, alpha_exp=2, beta_idx=2, beta_exp=2
            ),
        )
        assert g9

        # g10: g_+(a†_2 (a†_3)^3 a_2 (a_3)^3) = 2i a†_2(a†_3)^3 a_2(a_3)^3  — Geq, diagonal degree 8
        g10 = BosonicGenerator(
            kind="+",
            gamma=gamma_from_iotas_sum(
                n=12,
                alpha_indices=[2, 3],
                alpha_exps=[1, 3],
                beta_indices=[2, 3],
                beta_exps=[1, 3],
            ),
        )
        assert g10

        # g11: g_-((a†_0)^2 a†_1 a†_2 a†_3 a†_4 a_1 a_2 a_4)              — Gperp, degree 9
        g11 = BosonicGenerator(
            kind="-",
            gamma=gamma_from_iotas_sum(
                n=12,
                alpha_indices=[0, 2, 3, 4],
                alpha_exps=[2, 1, 1, 1],
                beta_indices=[1, 2, 3, 4],
            ),
        )
        assert g11

        generators = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]
        classification = mode_match(n=12, F=[fh], generators=generators)
        assert classification == DimensionResult.FINITE
        pass
