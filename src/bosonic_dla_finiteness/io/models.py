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
    label : human-readable name for the generator (e.g. "G1", "G2", etc.)
    description : optional longer description of the generator (e.g. "g_+(a†_0 a_1)")
    """

    kind: GeneratorKind
    alpha: list[int] = Field(..., description="Creation exponents")
    beta: list[int] = Field(..., description="Annihilation exponents")
    label: str = Field(..., description="Short label for the generator")
    description: str | None = Field(
        None, description="Longer description of the generator"
    )

    @model_validator(mode="after")
    def alpha_beta_same_length(self) -> GeneratorSpec:
        if len(self.alpha) != len(self.beta):
            raise ValueError(
                f"alpha (len={len(self.alpha)}) and beta (len={len(self.beta)}) "
                f"must have the same length."
            )
        return self

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
    The root configuration for the Hamiltonian system.
    """

    """
    Top-level input model for a fermionic DLA computation.

    Fields
    ------
    n_modes    : number of fermionic modes n
    omegas     : drift frequencies [omega_0, ..., omega_{n-1}]
    generators : list of g_+/g_- generator specifications

    Cross-field validation
    ----------------------
    - len(omegas) must equal n_modes
    - len(alpha) and len(beta) must equal n_modes for every generator
    """
    n_modes: Annotated[int, Field(ge=1)]
    omegas: list[float]
    generators: list[GeneratorSpec]
    free_hamiltonians: list[list[float]] | None = Field(
        None,
        description=(
            "Set F of free Hamiltonians X^(ℓ) = Σ_k x_k^(ℓ) (i a†_k a_k). "
            "Each entry is a coefficient vector of length n_modes. "
            "Defaults to [omegas] if omitted."
        ),
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> SystemConfig:
        if len(self.omegas) != self.n_modes:
            raise ValueError(
                f"Length of omegas ({len(self.omegas)}) must equal n_modes ({self.n_modes})."
            )

        # alpha / beta length for every generator
        for i, g in enumerate(self.generators):
            for vec, name in [(g.alpha, "alpha"), (g.beta, "beta")]:
                if len(vec) != self.n_modes:
                    raise ValueError(
                        f"generators[{i}].{name} has length {len(vec)} "
                        f"but n_modes={self.n_modes}."
                    )

        # free_hamiltonians: each vector must have length n_modes
        if self.free_hamiltonians is not None:
            for i, x in enumerate(self.free_hamiltonians):
                if len(x) != self.n_modes:
                    raise ValueError(
                        f"free_hamiltonians[{i}] has length {len(x)} "
                        f"but n_modes={self.n_modes}."
                    )

        return self

    def get_F(self) -> list[list[float]]:
        """Return the set F of free Hamiltonian coefficient vectors (defaults to [omegas])."""
        return (
            self.free_hamiltonians
            if self.free_hamiltonians is not None
            else [self.omegas]
        )

    @field_validator("generators")
    @classmethod
    def at_least_one_generator(
        cls, v: list[GeneratorSpec]
    ) -> list[GeneratorSpec]:
        if not v:
            raise ValueError("generators list must not be empty.")
        return v
