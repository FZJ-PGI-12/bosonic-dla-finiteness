"""
Tests for the `bosonic-dla` command-line entry point.

The CLI is the interface most users meet first, so its exit codes and printed
output are treated as behaviour worth pinning, not incidental formatting.
"""

import logging

import pytest

from bosonic_dla_finiteness.__main__ import main

CONFIG_DIR = "tests/configs"


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["bosonic-dla", *argv])
    main()


class TestExitCodes:
    def test_missing_file_exits_nonzero(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["bosonic-dla", "does_not_exist.yaml"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "Input file not found" in capsys.readouterr().err

    def test_valid_file_exits_cleanly(self, monkeypatch, capsys):
        _run(monkeypatch, [f"{CONFIG_DIR}/valid_example.yaml"])
        assert "Result:" in capsys.readouterr().out


class TestVerdictOutput:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("finiteness/finite/example1.yaml", "Result: Finite-dimensional"),
            ("finiteness/infinite/example1.yaml", "Result: Infinite"),
            ("finiteness/remaining/example1.yaml", "Result: Remaining"),
        ],
    )
    def test_each_verdict_is_printed(
        self, monkeypatch, capsys, path, expected
    ):
        _run(monkeypatch, [f"{CONFIG_DIR}/{path}"])
        assert expected in capsys.readouterr().out

    def test_remaining_lists_generators(self, monkeypatch, capsys):
        _run(
            monkeypatch,
            [f"{CONFIG_DIR}/finiteness/remaining/example1.yaml"],
        )
        out = capsys.readouterr().out
        assert "Remaining generators (" in out
        # Every listed generator is indented under the count line
        listed = [ln for ln in out.splitlines() if ln.startswith("  g_")]
        assert listed
        # Output is sorted, so a rerun is diffable
        assert listed == sorted(listed)

    def test_finite_does_not_list_generators(self, monkeypatch, capsys):
        _run(monkeypatch, [f"{CONFIG_DIR}/finiteness/finite/example1.yaml"])
        assert "Remaining generators" not in capsys.readouterr().out


class TestVerboseFlag:
    def test_verbose_enables_debug_logging(self, monkeypatch, caplog):
        with caplog.at_level(logging.DEBUG):
            _run(
                monkeypatch,
                ["-v", f"{CONFIG_DIR}/finiteness/finite/example1.yaml"],
            )
        assert any(r.levelno == logging.DEBUG for r in caplog.records)

    def test_long_form_verbose_accepted(self, monkeypatch, capsys):
        _run(
            monkeypatch,
            [
                "--verbose",
                f"{CONFIG_DIR}/finiteness/finite/example1.yaml",
            ],
        )
        assert "Result:" in capsys.readouterr().out


class TestArgumentParsing:
    def test_missing_config_argument_exits(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["bosonic-dla"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2  # argparse usage error
