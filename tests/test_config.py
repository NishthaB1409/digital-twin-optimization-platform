"""Config loading and validation.

A malformed line should fail at load time with a message naming the problem,
not halfway through a simulation with a KeyError.
"""

import copy

import pytest
import yaml

from dtmo.utils.config import (
    DEFAULT_CONFIG_PATH,
    FactoryConfig,
    load_config,
    parse_config,
)

MINIMAL = {
    "simulation": {"n_jobs": 10, "arrival_rate": 0.5, "processing_cv": 0.0, "seed": 1},
    "dispatch": {"weights": [1.0, 0.0, 0.0, 0.0]},
    "stations": [{"name": "A", "capacity": 1}, {"name": "B", "capacity": 2}],
    "families": [
        {
            "name": "F1",
            "mix": 1.0,
            "weight": 1.0,
            "due_factor": 2.0,
            "route": [
                {"station": "A", "mean_time": 3.0},
                {"station": "B", "mean_time": 4.0},
            ],
        }
    ],
}


@pytest.fixture
def raw():
    return copy.deepcopy(MINIMAL)


class TestShippedConfig:
    def test_default_config_exists_and_loads(self):
        assert DEFAULT_CONFIG_PATH.exists()
        assert isinstance(load_config(), FactoryConfig)

    def test_has_six_stations_and_five_families(self, config):
        assert len(config.stations) == 6
        assert len(config.families) == 5

    def test_every_route_targets_a_real_station(self, config):
        known = set(config.station_names)
        for family in config.families:
            for op in family.route:
                assert op.station in known

    def test_line_is_congested_but_stable(self, config):
        # Below ~0.7 the dispatch rule has no leverage; above 1.0 the queue
        # grows without bound and no rule can help. Day 1 tuning targets the
        # band in between, and this test pins that intent.
        loads = config.expected_utilisation()
        assert max(loads.values()) < 1.0, f"line is over capacity: {loads}"
        assert max(loads.values()) > 0.70, f"line is too lightly loaded: {loads}"

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config file not found"):
            load_config(tmp_path / "nope.yaml")


class TestParsing:
    def test_round_trips_through_yaml(self, raw, tmp_path):
        path = tmp_path / "f.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        assert load_config(path).station_names == ("A", "B")

    def test_reads_simulation_and_dispatch_blocks(self, raw):
        cfg = parse_config(raw)
        assert cfg.n_jobs == 10
        assert cfg.seed == 1
        assert cfg.dispatch_weights == (1.0, 0.0, 0.0, 0.0)

    def test_top_level_must_be_a_mapping(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_config(path)


class TestValidation:
    def test_route_through_unknown_station_is_rejected(self, raw):
        raw["families"][0]["route"][0]["station"] = "Nowhere"
        with pytest.raises(ValueError, match="unknown station"):
            parse_config(raw)

    def test_duplicate_station_names_are_rejected(self, raw):
        raw["stations"].append({"name": "A", "capacity": 1})
        with pytest.raises(ValueError, match="duplicate station"):
            parse_config(raw)

    def test_missing_required_key_names_the_key(self, raw):
        del raw["families"][0]["weight"]
        with pytest.raises(KeyError, match="weight"):
            parse_config(raw)

    def test_missing_stations_block_is_rejected(self, raw):
        del raw["stations"]
        with pytest.raises(KeyError, match="stations"):
            parse_config(raw)

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("n_jobs", 0, "n_jobs"),
            ("arrival_rate", 0.0, "arrival_rate"),
            ("processing_cv", -0.1, "processing_cv"),
        ],
    )
    def test_simulation_bounds(self, raw, field, value, match):
        raw["simulation"][field] = value
        with pytest.raises(ValueError, match=match):
            parse_config(raw)


class TestOverrides:
    def test_with_overrides_returns_a_changed_copy(self, config):
        changed = config.with_overrides(n_jobs=7, seed=99)
        assert (changed.n_jobs, changed.seed) == (7, 99)
        assert (config.n_jobs, config.seed) != (7, 99)

    def test_with_overrides_revalidates(self, config):
        with pytest.raises(ValueError, match="n_jobs"):
            config.with_overrides(n_jobs=0)

    def test_expected_utilisation_scales_with_arrival_rate(self, config):
        base = config.expected_utilisation()
        doubled = config.with_overrides(arrival_rate=config.arrival_rate * 2)
        for name, load in doubled.expected_utilisation().items():
            assert load == pytest.approx(base[name] * 2)

    def test_expected_utilisation_covers_every_station(self, config):
        assert set(config.expected_utilisation()) == set(config.station_names)
