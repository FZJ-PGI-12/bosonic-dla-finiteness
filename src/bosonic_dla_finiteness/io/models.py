"""
Pydantic models for the YAML input format: the generator specifications and
the top-level system configuration.

Validation is cross-field — SystemConfig.validate_dimensions checks every
omegas vector and every generator against n_modes, and expands the compact
`iotas` spelling into full exponent vectors of length n_modes.
"""

from __future__ import annotations

from typing import Annotated, cast

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from bosonic_dla_finiteness.operators.operator import (
    BosonicGenerator,
    GeneratorKind,
)

# GeneratorKind lives in operators.operator (the layer that defines the basis
# elements); it is re-exported here because it is part of the YAML schema.
__all__ = ["GeneratorKind", "GeneratorSpec", "IotasSpec", "SystemConfig"]


class IotasSpec(BaseModel):
    """Compact generator spec using mode indices instead of full exponent vectors."""

    alpha_indices: list[int] = Field(default_factory=list)
    alpha_exponents: list[int] | None = None
    beta_indices: list[int] = Field(default_factory=list)
    beta_exponents: list[int] | None = None

    @model_validator(mode="after")
    def check_exponent_lengths(self) -> IotasSpec:
        if self.alpha_exponents is not None and len(
            self.alpha_exponents
        ) != len(self.alpha_indices):
            raise ValueError(
                f"alpha_exponents (len={len(self.alpha_exponents)}) must match "
                f"alpha_indices (len={len(self.alpha_indices)})."
            )
        if self.beta_exponents is not None and len(self.beta_exponents) != len(
            self.beta_indices
        ):
            raise ValueError(
                f"beta_exponents (len={len(self.beta_exponents)}) must match "
                f"beta_indices (len={len(self.beta_indices)})."
            )
        return self

    def to_alpha_beta(self, n: int) -> tuple[list[int], list[int]]:
        """Expand to full exponent vectors of length n."""
        alpha = [0] * n
        beta = [0] * n
        a_exps = self.alpha_exponents or [1] * len(self.alpha_indices)
        b_exps = self.beta_exponents or [1] * len(self.beta_indices)
        for idx, exp in zip(self.alpha_indices, a_exps):
            alpha[idx] += exp
        for idx, exp in zip(self.beta_indices, b_exps):
            beta[idx] += exp
        return alpha, beta


class GeneratorSpec(BaseModel):
    """
    Specification of a single basis element of Â_n (see operators.operator):

        g_+^(α,β) = i(a^(β,α) + a^(α,β))
        g_-^(α,β) = a^(β,α) − a^(α,β)

    Note that g_- carries no factor of i: a^(β,α) − a^(α,β) is already
    skew-Hermitian, whereas i(a^(β,α) − a^(α,β)) would be Hermitian.

    Fields
    ------
    kind  : "+" for g_+, "-" for g_-
    alpha : creation exponents, (a†_0)^alpha_0 ... (a†_{n-1})^alpha_{n-1}
    beta  : annihilation exponents, (a_0)^beta_0 ... (a_{n-1})^beta_{n-1}
    iotas : compact alternative to alpha/beta using mode indices and exponents
    label : human-readable name for the generator (e.g. "G1", "G2", etc.)
    description : optional longer description of the generator
    """

    kind: GeneratorKind
    alpha: list[int] | None = Field(None, description="Creation exponents")
    beta: list[int] | None = Field(None, description="Annihilation exponents")
    iotas: IotasSpec | None = Field(
        None, description="Compact iota-based spec"
    )
    label: str = Field(..., description="Short label for the generator")
    description: str | None = Field(
        None, description="Longer description of the generator"
    )

    @model_validator(mode="after")
    def check_alpha_beta_or_iotas(self) -> GeneratorSpec:
        has_explicit = self.alpha is not None and self.beta is not None
        has_iotas = self.iotas is not None
        if not has_explicit and not has_iotas:
            raise ValueError("Specify either 'alpha'/'beta' or 'iotas'.")
        if has_explicit and has_iotas:
            raise ValueError(
                "Specify either 'alpha'/'beta' or 'iotas', not both."
            )
        if self.alpha is not None and self.beta is not None:
            if len(self.alpha) != len(self.beta):
                raise ValueError(
                    f"alpha (len={len(self.alpha)}) and beta "
                    f"(len={len(self.beta)}) must have the same length."
                )
        return self

    def _exponents(self) -> tuple[list[int], list[int]]:
        """
        Return (alpha, beta), both guaranteed non-None.

        check_alpha_beta_or_iotas admits only the explicit and the iotas
        spelling, and expand() fills alpha/beta in for the latter, so both are
        set by the time any accessor runs. Raising here keeps that reasoning
        checkable instead of implicit.
        """
        if self.alpha is None or self.beta is None:
            raise ValueError(
                "alpha/beta are unset; call expand(n) before accessing the "
                "exponent vectors of an iotas-specified generator."
            )
        return self.alpha, self.beta

    def expand(self, n: int) -> None:
        """Expand iotas to alpha/beta of length n and validate dimensions."""
        if self.iotas is not None:
            self.alpha, self.beta = self.iotas.to_alpha_beta(n)
        alpha, beta = self._exponents()
        for vec, name in [(alpha, "alpha"), (beta, "beta")]:
            if len(vec) != n:
                raise ValueError(
                    f"{name} has length {len(vec)} but expected n={n}."
                )

    @property
    def gamma(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return gamma = (alpha, beta) as a GammaIndex tuple."""
        alpha, beta = self._exponents()
        return (tuple(alpha), tuple(beta))

    @property
    def degree(self) -> int:
        """Total degree |gamma| = sum(alpha) + sum(beta)."""
        alpha, beta = self._exponents()
        return sum(alpha) + sum(beta)

    def to_generator(self) -> BosonicGenerator:
        """Convert to a BosonicGenerator (no coefficient)."""
        return BosonicGenerator(self.kind, self.gamma)


class SystemConfig(BaseModel):
    """
    Top-level input model for a bosonic DLA computation.

    Fields
    ------
    n_modes    : number of bosonic modes n
    omegas     : the set F of free Hamiltonians, either one drift-frequency
                 vector [omega_0, ..., omega_{n-1}] or a list of such vectors
    generators : list of g_+/g_- generator specifications

    Cross-field validation
    ----------------------
    - every vector in omegas must have length n_modes
    - len(alpha) and len(beta) must equal n_modes for every generator
    """

    n_modes: Annotated[int, Field(ge=1)]
    omegas: list[float] | list[list[float]] = Field(
        ...,
        description=(
            "Set F of free Hamiltonians X^(ℓ) = Σ_k ω_k^(ℓ) (i a†_k a_k). "
            "Either a single coefficient vector of length n_modes, or a list "
            "of such vectors for several free Hamiltonians."
        ),
    )
    generators: list[GeneratorSpec]

    @model_validator(mode="after")
    def validate_dimensions(self) -> SystemConfig:
        n = self.n_modes

        for i, x in enumerate(self.get_F()):
            if len(x) != n:
                raise ValueError(
                    f"omegas[{i}] has length {len(x)} but n_modes={n}."
                )

        for g in self.generators:
            g.expand(n)

        return self

    def get_F(self) -> list[list[float]]:
        """
        Return the set F as a list of coefficient vectors.

        Normalizes the single-vector spelling of `omegas` to [omegas], so
        callers never have to distinguish the two input shapes.
        """
        if self.omegas and isinstance(self.omegas[0], list):
            return self.omegas
        # Narrowing on omegas[0] tells mypy nothing about the empty case, which
        # the branch above excludes; the cast records that.
        return [cast("list[float]", self.omegas)]

    @field_validator("generators")
    @classmethod
    def at_least_one_generator(
        cls, v: list[GeneratorSpec]
    ) -> list[GeneratorSpec]:
        if not v:
            raise ValueError("generators list must not be empty.")
        return v
