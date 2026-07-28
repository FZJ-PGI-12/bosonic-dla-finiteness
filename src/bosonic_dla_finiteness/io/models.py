from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from bosonic_dla_finiteness.operators.operator import BosonicGenerator

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


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


class GeneratorKind(str, Enum):
    plus = "+"
    minus = "-"


class GeneratorSpec(BaseModel):
    """
    Specification of a single generator g_± = i*((a^gamma)† ± a^gamma).

    Fields
    ------
    kind  : "+" for g_+, "-" for g_-
    alpha : creation exponent vector  (a†_0)^alpha_0 ... (a†_n)^alpha_n
    beta  : annihilation exponent vector  (a_0)^beta_0 ... (a_n)^beta_n
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
        if has_explicit and len(self.alpha) != len(self.beta):
            raise ValueError(
                f"alpha (len={len(self.alpha)}) and beta (len={len(self.beta)}) "
                f"must have the same length."
            )
        return self

    def expand(self, n: int) -> None:
        """Expand iotas to alpha/beta of length n and validate dimensions."""
        if self.iotas is not None:
            self.alpha, self.beta = self.iotas.to_alpha_beta(n)
        for vec, name in [(self.alpha, "alpha"), (self.beta, "beta")]:
            if len(vec) != n:
                raise ValueError(
                    f"{name} has length {len(vec)} but expected n={n}."
                )

    @property
    def gamma(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return gamma = (alpha, beta) as a GammaIndex tuple."""
        return (tuple(self.alpha), tuple(self.beta))

    @property
    def degree(self) -> int:
        """Total degree |gamma| = sum(alpha) + sum(beta)."""
        return sum(self.alpha) + sum(self.beta)

    def to_generator(self) -> BosonicGenerator:
        """Convert to a BosonicGenerator (no coefficient)."""
        from bosonic_dla_finiteness.operators.operator import BosonicGenerator

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
        return [self.omegas]

    @field_validator("generators")
    @classmethod
    def at_least_one_generator(
        cls, v: list[GeneratorSpec]
    ) -> list[GeneratorSpec]:
        if not v:
            raise ValueError("generators list must not be empty.")
        return v
