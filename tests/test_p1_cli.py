"""The P1 commands dispatch to their own runners without changing v2 commands."""
from importlib import import_module

import pytest

from pgm_asc.cli import main


@pytest.mark.parametrize("command,module", [
    ("hierarchical", "hierarchical"),
    ("p1-objective", "p1_objectives"),
    ("p1-predictive", "p1_predictive"),
    ("p1-transition", "p1_transitions"),
    ("p1-brier", "p1_brier"),
])
def test_study_command_forwards_options(command, module, monkeypatch, tmp_path):
    calls = []
    runner = import_module(f"pgm_asc.{module}.run")
    monkeypatch.setattr(runner, "run", lambda *args: calls.append(args))
    config = tmp_path / "study.json"
    config.write_text('{"custom": true}')
    main([command, "fit", "--models", "gemma4b", "--datasets", "math",
          "--workers", "1", "--out", str(tmp_path), "--config", str(config), "--overwrite"])
    assert calls == [(["gemma4b"], ["math"], 1, str(tmp_path), "fit", True, {"custom": True})]


@pytest.mark.parametrize("stage", ["trace", "report"])
def test_stopping_command_dispatches(stage, monkeypatch):
    runner = import_module(f"pgm_asc.p1_stopping.{'run' if stage == 'trace' else 'report'}")
    calls = []
    monkeypatch.setattr(runner, "run" if stage == "trace" else "report", lambda *args: calls.append(args))
    main(["p1-stop", stage, "--workers", "1"])
    expected = [(None, None, 1, False)] if stage == "trace" else [()]
    assert calls == expected


@pytest.mark.parametrize("command", ["hierarchical", "p1-objective", "p1-predictive",
                                     "p1-transition", "p1-brier", "p1-stop"])
def test_p1_rejects_invalid_worker_count(command):
    with pytest.raises(SystemExit) as exc:
        main([command, "trace" if command == "p1-stop" else "fit", "--workers", "0"])
    assert exc.value.code == 2
