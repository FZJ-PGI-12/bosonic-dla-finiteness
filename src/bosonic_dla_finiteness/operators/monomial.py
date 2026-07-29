"""
monomial.py
-----------
Monomials indexed by gamma = (alpha, beta) in N^{2n}, representing the
normal-ordered monomial:

    a^gamma = (a†_0)^alpha_0 ... (a†_{n-1})^alpha_{n-1}
              (a_0)^beta_0  ... (a_{n-1})^beta_{n-1}

Mode indices are 0-based throughout.

The multi-degree mdeg(m) := gamma = (alpha, beta) specifies the frequencies
alpha_j and beta_j of a†_j and a_j respectively.

The degree of a monomial is |gamma| = sum_j (alpha_j + beta_j).

Special vectors:
    iota_k : n-vector with 1 at position k  (single-mode index)
    tau_p  : (iota_p, iota_p) in N^{2n}    (number operator a†_p a_p)
"""

from __future__ import annotations

# ── Type aliases ──────────────────────────────────────────────────────────────

MultiIndex = tuple[int, ...]  # alpha or beta, length n
GammaIndex = tuple[MultiIndex, MultiIndex]  # (alpha, beta)


# ── Special index constructors ────────────────────────────────────────────────


def iota(k: int, n: int) -> MultiIndex:
    """
    iota_k in N^n: unit vector with 1 at position k, 0 elsewhere.

    (iota_k)_p = delta_{kp}

    Example
    -------
    >>> iota(1, 3)
    (0, 1, 0)
    """
    v = [0] * n
    v[k] = 1
    return tuple(v)


def tau(p: int, n: int) -> GammaIndex:
    """
    tau_p in N^{2n}: (iota_p, iota_p).
    Represents the number operator a†_p a_p.

    Example
    -------
    >>> tau(1, 3)
    ((0, 1, 0), (0, 1, 0))
    """
    ip = iota(p, n)
    return (ip, ip)


def gamma_from_iotas(
    n: int,
    alpha_idx: int | None = None,
    beta_idx: int | None = None,
    alpha_exp: int = 1,
    beta_exp: int = 1,
) -> GammaIndex:
    """
    Construct a GammaIndex from single iota indices and optional exponents.

    alpha_exp / beta_exp set the power of the creation / annihilation operator
    at the given mode index. Omitted index defaults to the zero vector.

    Example
    -------
    >>> gamma_from_iotas(3, alpha_idx=0, alpha_exp=2)   # (a†_0)^2
    ((2, 0, 0), (0, 0, 0))
    """
    zero = tuple([0] * n)
    alpha = (
        tuple(alpha_exp * x for x in iota(alpha_idx, n))
        if alpha_idx is not None
        else zero
    )
    beta = (
        tuple(beta_exp * x for x in iota(beta_idx, n))
        if beta_idx is not None
        else zero
    )
    return (alpha, beta)


def gamma_from_iotas_sum(
    n: int,
    alpha_indices: list[int] | None = None,
    beta_indices: list[int] | None = None,
    alpha_exps: list[int] | None = None,
    beta_exps: list[int] | None = None,
) -> GammaIndex:
    """
    Construct a GammaIndex from lists of mode indices and optional exponents.

    alpha_exps / beta_exps are parallel to alpha_indices / beta_indices and set
    the exponent for each mode. Defaults to 1 for every index if omitted.

    Example
    -------
    >>> gamma_from_iotas_sum(4, beta_indices=[1, 3], beta_exps=[2, 1])  # a_1^2 a_3
    ((0, 0, 0, 0), (0, 2, 0, 1))
    """
    zero = tuple([0] * n)
    a_idx = alpha_indices or []
    b_idx = beta_indices or []
    a_exps = alpha_exps or [1] * len(a_idx)
    b_exps = beta_exps or [1] * len(b_idx)
    assert len(a_idx) == len(a_exps), (
        "alpha_indices and alpha_exps must have the same length"
    )
    assert len(b_idx) == len(b_exps), (
        "beta_indices and beta_exps must have the same length"
    )
    alpha = (
        tuple(
            sum(a_exps[k] * iota(i, n)[j] for k, i in enumerate(a_idx))
            for j in range(n)
        )
        if a_idx
        else zero
    )
    beta = (
        tuple(
            sum(b_exps[k] * iota(i, n)[j] for k, i in enumerate(b_idx))
            for j in range(n)
        )
        if b_idx
        else zero
    )
    return (alpha, beta)


def s_eq(gamma: GammaIndex) -> frozenset[int]:
    """S=(γ) = {k : α_k = β_k ≠ 0} — modes where the generator acts diagonally."""
    alpha, beta = gamma
    return frozenset(
        k for k, (a, b) in enumerate(zip(alpha, beta)) if a == b and a != 0
    )


def s_neq(gamma: GammaIndex) -> frozenset[int]:
    """S≠(γ) = {k : α_k ≠ β_k} — modes where the generator acts off-diagonally."""
    alpha, beta = gamma
    return frozenset(k for k, (a, b) in enumerate(zip(alpha, beta)) if a != b)


def gamma_degree(gamma: GammaIndex) -> int:
    """
    |gamma| = sum_j (alpha_j + beta_j).

    Example
    -------
    >>> gamma_degree(((1,0,0), (0,1,1)))
    3
    """
    alpha, beta = gamma
    return sum(alpha) + sum(beta)
