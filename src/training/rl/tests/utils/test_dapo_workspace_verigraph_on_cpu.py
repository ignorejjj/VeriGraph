import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

pytest.importorskip("ray")
pytest.importorskip("tensordict")

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _to_attr_dict(value):
    if isinstance(value, dict):
        return AttrDict({key: _to_attr_dict(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def test_prepare_workspace_skips_datamind_bootstrap_in_verigraph_mode():
    trainer = object.__new__(RayDAPOTrainer)
    trainer.config = _to_attr_dict(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "multi_turn": {
                        "verigraph": {
                            "enable": True,
                        }
                    }
                }
            },
            "data": {},
        }
    )

    assert trainer.prepare_workspace() is False


def test_prepare_workspace_is_noop_without_legacy_csv_workspace():
    trainer = object.__new__(RayDAPOTrainer)
    trainer.config = _to_attr_dict(
        {
            "data": {},
        }
    )

    assert trainer.prepare_workspace() is False


def test_prepare_workspace_requires_working_dir_when_legacy_csv_workspace_is_enabled():
    trainer = object.__new__(RayDAPOTrainer)
    trainer.config = _to_attr_dict(
        {
            "data": {
                "csv_folder": "/tmp/datamind_csvs",
            },
        }
    )

    with pytest.raises(ValueError, match="data.working_dir must be set when data.csv_folder is provided"):
        trainer.prepare_workspace()
