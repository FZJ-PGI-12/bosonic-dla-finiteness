from __future__ import annotations

"""
monomial.py
-----------
Monomials indexed by gamma = (alpha, beta) in N^{2n}, representing the
normal-ordered monomial:

    a^gamma = (a†_1)^alpha_1 ... (a†_n)^alpha_n (a_1)^beta_1 ... (a_n)^beta_n

The multi-degree mdeg(m) := gamma = (alpha, beta) specifies the frequencies
alpha_j and beta_j of a†_j and a_j respectively.

The degree of a monomial is |gamma| = sum_j (alpha_j + beta_j).

Special vectors:
    iota_k : n-vector with 1 at position k  (single-mode index)
    tau_p  : (iota_p, iota_p) in N^{2n}    (number operator a†_p a_p)
"""

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
    >>> gamma_from_iotas(12, alpha_idx=0, alpha_exp=2)
    ((2, 0, ..., 0), (0, 0, ..., 0))   # (a†_0)^2
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
    >>> gamma_from_iotas_sum(13, beta_indices=[10, 12], beta_exps=[100, 1])
    # a_10^100 · a_12
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


def zero_gamma(n: int) -> GammaIndex:
    """
    The zero multi-index in N^{2n}.
    Corresponds to the identity monomial a^0 = 1.
    """
    z = tuple([0] * n)
    return (z, z)


def s0(gamma: GammaIndex) -> frozenset[int]:
    """S0(γ) = {k : α_k = β_k = 0} — modes where the generator does not act."""
    alpha, beta = gamma
    return frozenset(
        k for k, (a, b) in enumerate(zip(alpha, beta)) if a == 0 and b == 0
    )


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


def gamma_modes(gamma: GammaIndex) -> frozenset[int]:
    """
    Set of mode indices j where alpha_j > 0 or beta_j > 0.

    Example
    -------
    >>> gamma_modes(((1,0,0), (0,1,1)))
    frozenset({0, 1, 2})
    """
    alpha, beta = gamma
    return frozenset(
        j for j, (a, b) in enumerate(zip(alpha, beta)) if a > 0 or b > 0
    )


def gamma_adjoint(gamma: GammaIndex) -> GammaIndex:
    """
    Adjoint of a^gamma: swap alpha and beta.

        (a^(alpha, beta))† = a^(beta, alpha)

    For a normal-ordered monomial, taking the adjoint reverses operator
    order and flips all daggers, which is equivalent to swapping alpha <-> beta.

    Example
    -------
    >>> gamma_adjoint(((1,0,0), (0,1,1)))
    ((0, 1, 1), (1, 0, 0))
    """
    alpha, beta = gamma
    return (beta, alpha)


def gamma_add(g1: GammaIndex, g2: GammaIndex) -> GammaIndex:
    """
    Component-wise addition: g1 + g2.
    Useful for tracking combined degree of operator products.
    """
    a1, b1 = g1
    a2, b2 = g2
    return (
        tuple(x + y for x, y in zip(a1, a2)),
        tuple(x + y for x, y in zip(b1, b2)),
    )


def is_self_adjoint_gamma(gamma: GammaIndex) -> bool:
    """True if gamma == gamma_adjoint(gamma), i.e. alpha == beta."""
    alpha, beta = gamma
    return alpha == beta


class BosonicMonomial:
    r"""
    Represents a normal-ordered monomial a^gamma = (a^\dagger)^alpha (a)^beta
    gamma = (alpha, beta) where alpha and beta are tuples of length n.
    """

    def __init__(self, gamma: GammaIndex) -> None:
        alpha, beta = gamma
        n = len(alpha)
        assert len(beta) == n, "alpha and beta must have the same length n."
        assert all(a >= 0 for a in alpha), (
            "alpha entries must be non-negative."
        )
        assert all(b >= 0 for b in beta), "beta entries must be non-negative."

        self._gamma = (tuple(alpha), tuple(beta))  # Store as (alpha, beta)

    @property
    def gamma(self) -> GammaIndex:
        """Return gamma = (alpha, beta)."""
        return self._gamma

    @property
    def alpha(self) -> MultiIndex:
        """Creation indices."""
        return self._gamma[0]

    @property
    def beta(self) -> MultiIndex:
        """Annihilation indices."""
        return self._gamma[1]

    def __eq__(self, other):
        if not isinstance(other, BosonicMonomial):
            return False
        return self.gamma == other.gamma

    def __hash__(self):
        return hash(self._gamma)

    def __repr__(self):
        return f"a^({self.alpha},{self.beta})"

    def dagger(self):
        # (a^\dagger)^alpha a^beta -> (a^\dagger)^beta a^alpha
        return BosonicMonomial((self.beta, self.alpha))
