"""
Tests for the table-based DLA classification helpers in dim_classifier.py.

Each test directly constructs a table (dict[Subspace, list[Cell]]) and a
decomposed-generator mapping so that the helper functions can be exercised
in isolation, independent of subspace-classification or chi_F logic.
"""

from bosonic_dla_finiteness.algebra.dim_classifier import (
    BgColor,
    Cell,
    DimensionResult,
    DotColor,
    _process_G1_G2core_Gom_GeqF,
    _process_Gperp_F,
)
from bosonic_dla_finiteness.algebra.subspaces import Subspace
from bosonic_dla_finiteness.operators.monomial import gamma_from_iotas
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


# ── _process_Gperp_F ─────────────────────────────────────────────────────────


class TestProcessGperpF:
    def test_empty_generators_returns_finite(self):
        assert (
            _process_Gperp_F(_table(3), _decomposed())
            == DimensionResult.FINITE
        )

    def test_s_neq_places_green_dot(self):
        # alpha=(2,1,0), beta=(1,0,0) → s_neq={0,1}, s_eq={}  →  Gperp
        n = 3
        table = _table(n)
        dec = _decomposed()
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((2, 1, 0), (1, 0, 0)))
        }
        assert _process_Gperp_F(table, dec) == DimensionResult.FINITE
        assert table[Subspace.Gperp_F][0].dot == DotColor.GREEN
        assert table[Subspace.Gperp_F][1].dot == DotColor.GREEN

    def test_s_neq_paints_correct_backgrounds(self):
        n = 3
        table = _table(n)
        dec = _decomposed()
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((2, 1, 0), (1, 0, 0)))
        }
        _process_Gperp_F(table, dec)
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
        assert _process_Gperp_F(table, dec) == DimensionResult.FINITE
        assert table[Subspace.Gperp_F][0].dot == DotColor.BLUE

    def test_s_eq_paints_blue_backgrounds(self):
        n = 4
        table = _table(n)
        dec = _decomposed()
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((1, 1, 1, 0), (1, 0, 0, 0)))
        }
        _process_Gperp_F(table, dec)
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
        assert _process_Gperp_F(table, dec) == DimensionResult.INFINITE

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
        assert _process_Gperp_F(table, dec) == DimensionResult.INFINITE

    def test_backgrounds_not_overwritten(self):
        # A pre-existing background must not be changed by the helper.
        n = 3
        table = _table(n)
        table[Subspace.G0][0].bg = BgColor.GREEN  # pre-existing, not RED

        dec = _decomposed()
        dec[Subspace.Gperp_F] = {
            BosonicGenerator(kind="-", gamma=((2, 1, 0), (1, 0, 0)))
        }
        _process_Gperp_F(table, dec)
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
