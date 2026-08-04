# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are derived from git tags by setuptools-scm, so each released version
below corresponds to a `vX.Y.Z` tag.

## [Unreleased]

## [1.0.0] - 2026-08-04

First stable release. The public API is `check_finiteness`, `FinitenessResult`,
`DimensionResult`, `BosonicGenerator`, `FreeHamiltonian`, `Subspace`, the
multi-index constructors in `operators.monomial`, and the YAML schema in
`io.models`. Incompatible changes to these will come with a major version bump.

### Added

- `check_finiteness`: table-based classification of a bosonic DLA as
  finite-dimensional, provably infinite-dimensional, or inconclusive
  (`REMAINING`, with the generators requiring further analysis returned).
- Subspace decomposition of the skew-Hermitian Weyl algebra
  $\hat{A}_n = G^0 \oplus G^1 \oplus G^2 \oplus G^= \oplus G^{om} \oplus G^\perp$,
  and the refinement into $G^2_F$, $G^2_{core}$, $G^=_F$, $G^\perp_F$ via
  $\chi_F$.
- `FreeHamiltonian` with span reduction (`compute_F_prime`) and the $\chi_F$ map.
- `BosonicGenerator` basis elements $g_\pm^\gamma$ with canonical-index
  validation, and multi-index constructors `gamma_from_iotas`,
  `gamma_from_iotas_sum`.
- YAML input format with pydantic validation, supporting explicit
  `alpha`/`beta` vectors or the compact `iotas` spelling, and one or several
  free Hamiltonians in `omegas`.
- `bosonic-dla` command-line entry point with `-v/--verbose` logging.
- `py.typed` marker: the package ships its annotations to downstream type
  checkers (PEP 561).

### Known limitations

- A `REMAINING` verdict is returned whenever more than one generator lies in
  $\langle G^\perp_F \cup T(G^2_F) \rangle$. Only the trivial single-generator
  case of Step 3 is resolved; the general case requires further analysis.
- `ZERO_TOL` is an absolute tolerance ($10^{-12}$) and therefore presumes
  frequencies of order unity. Frequency units for which typical values are very
  small make the tolerance significant, and modes merely close in frequency may
  then be treated as exactly degenerate.
