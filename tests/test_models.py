from pathlib import Path

import pytest

from bosonic_dla_finiteness.io.loader import load_from_yaml
from bosonic_dla_finiteness.io.models import (
    GeneratorKind,
    GeneratorSpec,
    IotasSpec,
    SystemConfig,
)


@pytest.fixture
def configs_dir():
    """Path to the test configs directory."""
    return Path(__file__).parent / "configs"


@pytest.fixture
def valid_example_path(configs_dir):
    """Path to the valid example YAML config."""
    return configs_dir / "valid_example.yaml"


@pytest.fixture
def minimal_config_path(configs_dir):
    """Path to the minimal valid YAML config."""
    return configs_dir / "minimal.yaml"


@pytest.fixture
def invalid_omegas_length_path(configs_dir):
    """Path to invalid config (omegas length mismatch)."""
    return configs_dir / "invalid_omegas_length.yaml"


@pytest.fixture
def invalid_alpha_length_path(configs_dir):
    """Path to invalid config (alpha length mismatch)."""
    return configs_dir / "invalid_alpha_length.yaml"


class TestLoadFromYAML:
    """Tests loading and validating YAML configurations."""

    def test_load_valid_example(self, valid_example_path):
        """Test loading the valid example config."""
        config = load_from_yaml(valid_example_path)
        assert isinstance(config, SystemConfig)

    def test_load_minimal_config(self, minimal_config_path):
        """Test loading minimal valid config."""
        config = load_from_yaml(minimal_config_path)
        assert isinstance(config, SystemConfig)
        assert config.n_modes == 1
        assert config.omegas == [2.5]
        assert len(config.generators) == 1

    def test_yaml_n_modes(self, valid_example_path):
        """Test that n_modes is correctly loaded."""
        config = load_from_yaml(valid_example_path)
        assert config.n_modes == 3

    def test_yaml_omegas(self, valid_example_path):
        """Test that omegas are correctly loaded."""
        config = load_from_yaml(valid_example_path)
        assert config.omegas == [1.0, 2.0, 3.0]
        assert len(config.omegas) == config.n_modes

    def test_yaml_generators_count(self, valid_example_path):
        """Test that all generators are loaded."""
        config = load_from_yaml(valid_example_path)
        assert len(config.generators) == 6

    def test_yaml_generator_kinds(self, valid_example_path):
        """Test that generator kinds are correct."""
        config = load_from_yaml(valid_example_path)
        kinds = [g.kind for g in config.generators]
        assert kinds == [
            GeneratorKind.plus,
            GeneratorKind.minus,
            GeneratorKind.plus,
            GeneratorKind.minus,
            GeneratorKind.plus,
            GeneratorKind.minus,
        ]

    def test_yaml_generator_labels(self, valid_example_path):
        """Test that generator labels match expected values."""
        config = load_from_yaml(valid_example_path)
        labels = [g.label for g in config.generators]
        assert labels == ["G1", "G2", "G3", "G4", "G5", "G6"]

    def test_yaml_generator_alpha_beta_match_n_modes(self, valid_example_path):
        """Test that all alpha and beta vectors match n_modes."""
        config = load_from_yaml(valid_example_path)
        for i, gen in enumerate(config.generators):
            assert len(gen.alpha) == config.n_modes, (
                f"Generator {i}: alpha length mismatch"
            )
            assert len(gen.beta) == config.n_modes, (
                f"Generator {i}: beta length mismatch"
            )

    def test_yaml_first_generator_structure(self, valid_example_path):
        """Test structure of the first generator (G1)."""
        config = load_from_yaml(valid_example_path)
        g1 = config.generators[0]

        assert g1.kind == GeneratorKind.plus
        assert g1.alpha == [1, 0, 0]
        assert g1.beta == [0, 1, 0]
        assert g1.label == "G1"
        assert g1.description == "g_+(a†_0 a_1)"

    def test_yaml_generator_degrees(self, valid_example_path):
        """Test that generator degrees are calculated correctly."""
        config = load_from_yaml(valid_example_path)

        # G1-G4: degree 2 (hopping terms)
        for i in range(4):
            assert config.generators[i].degree == 2

        # G5-G6: degree 3 (three-body terms)
        for i in range(4, 6):
            assert config.generators[i].degree == 3

    def test_invalid_omegas_length(self, invalid_omegas_length_path):
        """Test that invalid omegas length raises ValidationError."""
        with pytest.raises(ValueError):
            load_from_yaml(invalid_omegas_length_path)

    def test_invalid_alpha_length(self, invalid_alpha_length_path):
        """Test that invalid alpha length raises ValidationError."""
        with pytest.raises(ValueError):
            load_from_yaml(invalid_alpha_length_path)

    def test_yaml_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_from_yaml("nonexistent_file.yaml")


class TestIotasSpecValidation:
    """Exponent lists, when given, must be parallel to their index lists."""

    def test_alpha_exponent_length_mismatch(self):
        with pytest.raises(ValueError, match="alpha_exponents"):
            IotasSpec(alpha_indices=[0, 1], alpha_exponents=[2])

    def test_beta_exponent_length_mismatch(self):
        with pytest.raises(ValueError, match="beta_exponents"):
            IotasSpec(beta_indices=[0], beta_exponents=[1, 1])

    def test_omitted_exponents_default_to_one(self):
        alpha, beta = IotasSpec(
            alpha_indices=[0, 2], beta_indices=[1]
        ).to_alpha_beta(3)
        assert alpha == [1, 0, 1]
        assert beta == [0, 1, 0]

    def test_repeated_index_accumulates(self):
        alpha, _ = IotasSpec(alpha_indices=[1, 1]).to_alpha_beta(3)
        assert alpha == [0, 2, 0]


class TestGeneratorSpecValidation:
    def test_neither_alpha_beta_nor_iotas_rejected(self):
        with pytest.raises(ValueError, match="Specify either"):
            GeneratorSpec(kind="+", label="G")

    def test_both_alpha_beta_and_iotas_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            GeneratorSpec(
                kind="+",
                alpha=[1, 0],
                beta=[0, 1],
                iotas=IotasSpec(alpha_indices=[0]),
                label="G",
            )

    def test_alpha_beta_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            GeneratorSpec(kind="+", alpha=[1, 0, 0], beta=[0, 1], label="G")

    def test_expand_rejects_wrong_length(self):
        spec = GeneratorSpec(kind="+", alpha=[1, 0], beta=[0, 1], label="G")
        with pytest.raises(ValueError, match="expected n=3"):
            spec.expand(3)

    def test_accessors_require_expand_for_iotas_spec(self):
        """
        An iotas-only spec has no exponent vectors until expand(n) runs. The
        guard names that instead of raising TypeError from len(None).
        """
        spec = GeneratorSpec(
            kind="+", iotas=IotasSpec(alpha_indices=[0]), label="G"
        )
        with pytest.raises(ValueError, match="call expand"):
            _ = spec.gamma

    def test_degree_after_expand(self):
        spec = GeneratorSpec(
            kind="+",
            iotas=IotasSpec(alpha_indices=[0], alpha_exponents=[2]),
            label="G",
        )
        spec.expand(3)
        assert spec.degree == 2
        assert spec.gamma == ((2, 0, 0), (0, 0, 0))


class TestSystemConfigGeneratorList:
    def test_empty_generators_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SystemConfig(n_modes=2, omegas=[1.0, 1.0], generators=[])
