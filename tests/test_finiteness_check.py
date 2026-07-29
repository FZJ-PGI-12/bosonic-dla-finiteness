"""
Tests for the table-based DLA classification helpers in finiteness_check.py.

Each test directly constructs a table (dict[Subspace, list[Cell]]) and a
decomposed-generator mapping so that the helper functions can be exercised
in isolation, independent of subspace-classification or chi_F logic.
"""

from pathlib import Path

import pytest

from bosonic_dla_finiteness.algebra.finiteness_check import (
    BgColor,
    Cell,
    DimensionResult,
    DotColor,
    _has_orange_green_conflict,
    _process_G1_G2core_Gom_GeqF,
    _process_GperpF,
    check_finiteness,
)
from bosonic_dla_finiteness.algebra.free_hamiltonian import FreeHamiltonian
from bosonic_dla_finiteness.algebra.subspaces import Subspace
from bosonic_dla_finiteness.constants import ZERO_TOL
from bosonic_dla_finiteness.io.loader import load_from_yaml
from bosonic_dla_finiteness.operators.monomial import (
    gamma_from_iotas,
    gamma_from_iotas_sum,
)
from bosonic_dla_finiteness.operators.operator import BosonicGenerator

# ── Helpers ──────────────────────────────────────────────────────────────────

_ROWS = [
    Subspace.G0,
    Subspace.G1,
    Subspace.G2_core,
    Subspace.Gom,
    Subspace.Geq_F,
    Subspace.G2_F,
    Subspace.Gperp_F,
]


def _table(n: int) -> dict[Subspace, list[Cell]]:
    return {row: [Cell() for _ in range(n)] for row in _ROWS}


def _decomposed() -> dict[Subspace, set[BosonicGenerator]]:
    return {s: set() for s in Subspace}


# ── _process_GperpF ─────────────────────────────────────────────────────────


class TestProcessGperpF:
    def test_empty_generators_returns_finite(self):
        assert (
            _process_GperpF(_table(3), _decomposed()) == DimensionResult.FINITE
        )

    def test_s_neq_places_green_dot(self):
        # alpha=(2,1,0), beta=(1,0,0) → s_neq={0,1}, s_eq={}  →  Gperp
        n = 3
        table = _table(n)
        dec = _decomposed()
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((2, 1, 0), (1, 0, 0)))
        }
        assert _process_GperpF(table, dec) == DimensionResult.FINITE
        assert table[Subspace.Gperp_F][0].dot == DotColor.GREEN
        assert table[Subspace.Gperp_F][1].dot == DotColor.GREEN

    def test_s_neq_paints_correct_backgrounds(self):
        n = 3
        table = _table(n)
        dec = _decomposed()
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((2, 1, 0), (1, 0, 0)))
        }
        _process_GperpF(table, dec)
        # column 0: RED for G0/G1/G2_core/Gom/Geq_F, ORANGE for G2_F, GREEN for Gperp_F
        assert table[Subspace.G0][0].bg == BgColor.RED
        assert table[Subspace.G1][0].bg == BgColor.RED
        assert table[Subspace.G2_core][0].bg == BgColor.RED
        assert table[Subspace.Gom][0].bg == BgColor.RED
        assert table[Subspace.Geq_F][0].bg == BgColor.RED
        assert table[Subspace.G2_F][0].bg == BgColor.ORANGE
        assert table[Subspace.Gperp_F][0].bg == BgColor.GREEN

    def test_s_eq_places_blue_dot(self):
        # alpha=(1,1,1,0), beta=(1,0,0,0) → s_eq={0}, s_neq={1,2}  →  Gperp
        n = 4
        table = _table(n)
        dec = _decomposed()
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((1, 1, 1, 0), (1, 0, 0, 0)))
        }
        assert _process_GperpF(table, dec) == DimensionResult.FINITE
        assert table[Subspace.Gperp_F][0].dot == DotColor.BLUE

    def test_s_eq_paints_blue_backgrounds(self):
        n = 4
        table = _table(n)
        dec = _decomposed()
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((1, 1, 1, 0), (1, 0, 0, 0)))
        }
        _process_GperpF(table, dec)
        for row in [
            Subspace.G1,
            Subspace.G2_core,
            Subspace.Gom,
            Subspace.Geq_F,
            Subspace.G2_F,
            Subspace.Gperp_F,
        ]:
            assert table[row][0].bg == BgColor.BLUE

    def test_green_on_blue_returns_infinite(self):
        # Seed the table: (Gperp_F, 0) already BLUE
        # Then process a generator with s_neq∋0 → GREEN on BLUE → INFINITE
        n = 4
        table = _table(n)
        table[Subspace.Gperp_F][0].bg = BgColor.BLUE

        dec = _decomposed()
        # alpha=(2,1,0,0), beta=(1,0,0,0) → s_neq={0,1}
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((2, 1, 0, 0), (1, 0, 0, 0)))
        }
        assert _process_GperpF(table, dec) == DimensionResult.INFINITE

    def test_blue_on_green_returns_infinite(self):
        # Seed the table: (Gperp_F, 0) already GREEN
        # Then process a generator with s_eq∋0 → BLUE on GREEN → INFINITE
        n = 4
        table = _table(n)
        table[Subspace.Gperp_F][0].bg = BgColor.GREEN

        dec = _decomposed()
        # alpha=(1,1,1,0), beta=(1,0,0,0) → s_eq={0}
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((1, 1, 1, 0), (1, 0, 0, 0)))
        }
        assert _process_GperpF(table, dec) == DimensionResult.INFINITE

    def test_backgrounds_not_overwritten(self):
        # A pre-existing background must not be changed by the helper.
        n = 3
        table = _table(n)
        table[Subspace.G0][0].bg = BgColor.GREEN  # pre-existing, not RED

        dec = _decomposed()
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((2, 1, 0), (1, 0, 0)))
        }
        _process_GperpF(table, dec)
        assert table[Subspace.G0][0].bg == BgColor.GREEN  # must be unchanged


# ── _process_G1_G2core_Gom_GeqF ──────────────────────────────────────────────


class TestProcessG1G2coreGomGeqF:
    def test_empty_generators_returns_finite(self):
        assert (
            _process_G1_G2core_Gom_GeqF(_table(3), _decomposed())
            == DimensionResult.FINITE
        )

    def test_g1_s_neq_places_green_dot_and_backgrounds(self):
        n = 3
        table = _table(n)
        dec = _decomposed()
        # G1: alpha=(1,0,0), beta=(0,0,0) → s_neq={0}
        dec[Subspace.G1] = {
            BosonicGenerator(
                kind="-", gamma=gamma_from_iotas(n=3, alpha_idx=0)
            )
        }
        assert (
            _process_G1_G2core_Gom_GeqF(table, dec) == DimensionResult.FINITE
        )
        assert table[Subspace.G1][0].dot == DotColor.GREEN
        for row in [
            Subspace.G1,
            Subspace.G2_core,
            Subspace.Gom,
            Subspace.Geq_F,
            Subspace.G2_F,
        ]:
            assert table[row][0].bg == BgColor.GREEN

    def test_g1_s_neq_on_blue_returns_infinite(self):
        n = 3
        table = _table(n)
        table[Subspace.G1][0].bg = BgColor.BLUE
        dec = _decomposed()
        dec[Subspace.G1] = {
            BosonicGenerator(
                kind="-", gamma=gamma_from_iotas(n=3, alpha_idx=0)
            )
        }
        assert (
            _process_G1_G2core_Gom_GeqF(table, dec) == DimensionResult.INFINITE
        )

    def test_g1_s_neq_on_red_returns_infinite(self):
        n = 3
        table = _table(n)
        table[Subspace.G1][0].bg = BgColor.RED
        dec = _decomposed()
        dec[Subspace.G1] = {
            BosonicGenerator(
                kind="-", gamma=gamma_from_iotas(n=3, alpha_idx=0)
            )
        }
        assert (
            _process_G1_G2core_Gom_GeqF(table, dec) == DimensionResult.INFINITE
        )

    def test_geq_f_s_eq_places_blue_dot_and_backgrounds(self):
        n = 3
        table = _table(n)
        dec = _decomposed()
        # Geq: alpha=beta=(2,0,0) → s_eq={0}, degree=4
        dec[Subspace.Geq_F] = {
            BosonicGenerator(kind="+", gamma=((2, 0, 0), (2, 0, 0)))
        }
        assert (
            _process_G1_G2core_Gom_GeqF(table, dec) == DimensionResult.FINITE
        )
        assert table[Subspace.Geq_F][0].dot == DotColor.BLUE
        for row in [
            Subspace.G1,
            Subspace.G2_core,
            Subspace.Gom,
            Subspace.Geq_F,
            Subspace.G2_F,
        ]:
            assert table[row][0].bg == BgColor.BLUE

    def test_geq_f_s_eq_on_red_returns_infinite(self):
        n = 3
        table = _table(n)
        table[Subspace.Geq_F][0].bg = BgColor.RED
        dec = _decomposed()
        dec[Subspace.Geq_F] = {
            BosonicGenerator(kind="+", gamma=((2, 0, 0), (2, 0, 0)))
        }
        assert (
            _process_G1_G2core_Gom_GeqF(table, dec) == DimensionResult.INFINITE
        )

    def test_geq_f_s_eq_on_green_returns_infinite(self):
        n = 3
        table = _table(n)
        table[Subspace.Geq_F][0].bg = BgColor.GREEN
        dec = _decomposed()
        dec[Subspace.Geq_F] = {
            BosonicGenerator(kind="+", gamma=((2, 0, 0), (2, 0, 0)))
        }
        assert (
            _process_G1_G2core_Gom_GeqF(table, dec) == DimensionResult.INFINITE
        )

    def test_backgrounds_not_overwritten(self):
        # Background set by an earlier step must not be replaced.
        n = 3
        table = _table(n)
        table[Subspace.G2_F][
            0
        ].bg = BgColor.ORANGE  # set by Gperp_F step earlier

        dec = _decomposed()
        dec[Subspace.G1] = {
            BosonicGenerator(
                kind="-", gamma=gamma_from_iotas(n=3, alpha_idx=0)
            )
        }
        _process_G1_G2core_Gom_GeqF(table, dec)
        assert (
            table[Subspace.G2_F][0].bg == BgColor.ORANGE
        )  # must be unchanged


# ── _has_orange_green_conflict ────────────────────────────────────────────────


def _adj(n: int) -> list[list[int]]:
    return [[0] * n for _ in range(n)]


def _dotted_cell(bg: BgColor) -> Cell:
    return Cell(bg=bg, dot=DotColor.GREEN)


class TestHasOrangeGreenConflict:
    def test_no_dots_returns_false(self):
        n = 3
        assert not _has_orange_green_conflict(_adj(n), _table(n))

    def test_single_isolated_orange_dot_returns_false(self):
        n = 3
        table = _table(n)
        table[Subspace.G2_F][0] = _dotted_cell(BgColor.ORANGE)
        assert not _has_orange_green_conflict(_adj(n), table)

    def test_single_isolated_green_dot_returns_false(self):
        n = 3
        table = _table(n)
        table[Subspace.G2_F][0] = _dotted_cell(BgColor.GREEN)
        assert not _has_orange_green_conflict(_adj(n), table)

    def test_two_connected_orange_dots_returns_false(self):
        n = 3
        table = _table(n)
        table[Subspace.G2_F][0] = _dotted_cell(BgColor.ORANGE)
        table[Subspace.G2_F][1] = _dotted_cell(BgColor.ORANGE)
        adj = _adj(n)
        adj[0][1] = adj[1][0] = 1
        assert not _has_orange_green_conflict(adj, table)

    def test_directly_connected_orange_and_green_returns_true(self):
        n = 3
        table = _table(n)
        table[Subspace.G2_F][0] = _dotted_cell(BgColor.ORANGE)
        table[Subspace.G2_F][1] = _dotted_cell(BgColor.GREEN)
        adj = _adj(n)
        adj[0][1] = adj[1][0] = 1
        assert _has_orange_green_conflict(adj, table)

    def test_orange_and_green_in_separate_components_returns_false(self):
        # 0(orange) -- 1(orange)    2(green) -- 3(green): no cross-component path
        n = 4
        table = _table(n)
        table[Subspace.G2_F][0] = _dotted_cell(BgColor.ORANGE)
        table[Subspace.G2_F][1] = _dotted_cell(BgColor.ORANGE)
        table[Subspace.G2_F][2] = _dotted_cell(BgColor.GREEN)
        table[Subspace.G2_F][3] = _dotted_cell(BgColor.GREEN)
        adj = _adj(n)
        adj[0][1] = adj[1][0] = 1
        adj[2][3] = adj[3][2] = 1
        assert not _has_orange_green_conflict(adj, table)

    def test_orange_reaches_green_through_intermediate_dot(self):
        # 0(orange) -- 1(no bg) -- 2(green): path exists through intermediate
        n = 3
        table = _table(n)
        table[Subspace.G2_F][0] = _dotted_cell(BgColor.ORANGE)
        table[Subspace.G2_F][1] = Cell(bg=None, dot=DotColor.GREEN)
        table[Subspace.G2_F][2] = _dotted_cell(BgColor.GREEN)
        adj = _adj(n)
        adj[0][1] = adj[1][0] = 1
        adj[1][2] = adj[2][1] = 1
        assert _has_orange_green_conflict(adj, table)


_CONFIGS_DIR = Path(__file__).parent / "configs/finiteness"
_EXPECTED = {
    "finite": DimensionResult.FINITE,
    "infinite": DimensionResult.INFINITE,
    "remaining": DimensionResult.REMAINING,
}


def _collect_cases():
    return [
        pytest.param(
            path,
            _EXPECTED[path.parent.name],
            id=f"{path.parent.name}/{path.stem}",
        )
        for path in sorted(_CONFIGS_DIR.rglob("*.yaml"))
        if path.parent.name in _EXPECTED
    ]


@pytest.mark.parametrize("yaml_path,expected", _collect_cases())
def test_finiteness_from_yaml(yaml_path, expected):
    config = load_from_yaml(yaml_path)
    generators = [g.to_generator() for g in config.generators]
    result = check_finiteness(
        n=config.n_modes,
        F=[FreeHamiltonian(x) for x in config.get_F()],
        generators=generators,
    )
    assert result.dimension == expected


class TestFinitenessCheck:
    @pytest.fixture
    def free_hamiltonians(self):
        """List of free Hamiltonians for testing."""
        return [FreeHamiltonian([i for i in range(1, 13)])]

    @pytest.fixture
    def generators(self):
        """List of generators for testing."""
        # g1: g_+(a†_10)         = i(a†_10 + a_10)                             — G1, linear on mode 0
        g1 = BosonicGenerator(
            kind="+", gamma=gamma_from_iotas(n=12, alpha_idx=10)
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
        return generators

    def test_finiteness_check_with_generators(
        self, free_hamiltonians, generators
    ):
        result = check_finiteness(
            n=12, F=free_hamiltonians, generators=generators
        )
        assert result.dimension == DimensionResult.FINITE


class TestPositiveFrequencies:
    """
    All frequencies must satisfy omega_k > 0.
    """

    # (a†_0)^2 — one off-diagonal mode; chi_F = 2*omega_0 vanishes at omega_0=0
    SQUEEZE_ONE_MODE = ((2, 0), (0, 0))
    # a†_0 a†_1 — two off-diagonal modes, but not of the form (ι_p, ι_q);
    # chi_F = omega_0 + omega_1 vanishes at omega = (1, -1)
    SQUEEZE_TWO_MODE = ((1, 1), (0, 0))

    @pytest.mark.parametrize(
        "omegas,gamma",
        [
            pytest.param([0.0, 1.0], SQUEEZE_ONE_MODE, id="zero-frequency"),
            pytest.param([1.0, -1.0], SQUEEZE_TWO_MODE, id="opposite-signs"),
            pytest.param([-1.0, -2.0], SQUEEZE_ONE_MODE, id="all-negative"),
            # Positive, but at or below ZERO_TOL: chi_F is then judged to
            # vanish, so the squeezing generators reach G^2_F regardless.
            pytest.param([1e-15, 1.0], SQUEEZE_ONE_MODE, id="tiny-frequency"),
            pytest.param(
                [1e-13, 1e-13], SQUEEZE_TWO_MODE, id="both-tiny-frequencies"
            ),
            pytest.param(
                [ZERO_TOL, 1.0], SQUEEZE_ONE_MODE, id="exactly-at-tolerance"
            ),
        ],
    )
    def test_non_positive_frequencies_rejected(self, omegas, gamma):
        gen = BosonicGenerator(kind="+", gamma=gamma)
        with pytest.raises(ValueError, match="not positive"):
            check_finiteness(
                n=2, F=[FreeHamiltonian(omegas)], generators=[gen]
            )

    def test_positive_frequencies_accepted(self):
        """The same squeezing generators are fine once omega_k > 0."""
        gens = [
            BosonicGenerator(kind="+", gamma=self.SQUEEZE_ONE_MODE),
            BosonicGenerator(kind="+", gamma=self.SQUEEZE_TWO_MODE),
        ]
        result = check_finiteness(
            n=2, F=[FreeHamiltonian([1.0, 2.0])], generators=gens
        )
        assert result.dimension in set(DimensionResult)

    def test_rejection_precedes_classification(self):
        """
        A non-positive frequency is rejected even when no generator would
        exercise the G^2_F path, so the hypothesis is enforced uniformly.
        """
        gen = BosonicGenerator(kind="+", gamma=((1, 0), (0, 0)))  # G1
        with pytest.raises(ValueError, match="not positive"):
            check_finiteness(
                n=2, F=[FreeHamiltonian([1.0, 0.0])], generators=[gen]
            )
