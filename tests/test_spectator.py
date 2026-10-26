"""Spectator timeline: recording, rebuilding any step, and saving a session."""
import numpy as np
import pytest

from nasal import spectator


def timeline():
    original = np.zeros((4, 5, 3), dtype=np.int8)
    original[1:3, 1:4, 1] = 3
    tl = spectator.Timeline("P900", original, np.array([0.5, 0.5, 1.5]), np.array([1.0, 2.0, 3.0]), "test")
    tl.add("you", "open", "opened P900", result={"total_resistance": 0.3, "left_resistance": 0.6,
                                                 "right_resistance": 0.6, "cells": 6})
    wider = original.copy()
    wider[0, 1:4, 1] = 3
    tl.add("surgeon agent", "operate", "widen", tags=wider, detail={"operation": {"technique": "widen"}})
    tl.add("surgeon agent", "says", "checking it")
    tl.add("surgeon agent", "result", "OpenFOAM result", tags=wider,
           result={"total_resistance": 0.25, "left_resistance": 0.5, "right_resistance": 0.5})
    tl.add("surgeon agent", "operate", "undo widen", tags=original)
    return tl, original, wider


def test_any_step_is_rebuilt_from_the_original_and_its_changes():
    tl, original, wider = timeline()
    assert [e["step"] for e in tl.events] == [0, 1, 2, 3, 4]
    assert np.array_equal(tl.tags_at(0), original)
    assert np.array_equal(tl.tags_at(2), wider)  # a step without a change shows the last airway
    assert np.array_equal(tl.tags_at(4), original)
    assert len(tl.events[1]["diff"][0]) == 3  # only the changed voxels are stored


def test_result_belongs_to_the_airway_it_was_computed_on():
    tl, _, _ = timeline()
    assert tl.result_at(0)["total_resistance"] == 0.3
    assert "cells" not in tl.result_at(0)
    assert tl.result_at(1) is None and tl.result_at(2) is None  # changed, not simulated yet
    assert tl.result_at(3)["total_resistance"] == 0.25
    assert tl.result_at(4) is None


def test_changes_since_the_original_or_of_one_step():
    tl, _, _ = timeline()
    since = tl.changes_at(2)
    assert len(since["added"]) == 3 and len(since["removed"]) == 0
    assert since["added"][0] == pytest.approx([1.0, 2.5, 4.5])  # origin + index * spacing, mm
    assert len(tl.changes_at(2, since_original=False)["added"]) == 0
    assert len(tl.changes_at(4, since_original=False)["removed"]) == 3
    assert len(tl.changes_at(4)["added"]) == 0


def test_table_series_and_detail():
    tl, _, _ = timeline()
    table = tl.table()
    assert list(table["who"]) == ["you"] + ["surgeon agent"] * 4
    assert table["total resistance"].tolist() == [0.3, "", "", 0.25, ""]
    assert tl.resistance_series() == ([0, 3], [0.3, 0.25])
    text = spectator.detail_markdown(tl.events[1])
    assert "surgeon agent" in text and '"technique": "widen"' in text
    assert len(spectator.series_figure(tl, 2).data) == 1


def test_save_and_load_keep_the_whole_session(tmp_path):
    tl, original, wider = timeline()
    path = tl.save(tmp_path / "session.npz")
    back = spectator.Timeline.load(path)
    assert back.name == "P900" and back.how == "test" and len(back.events) == 5
    assert np.array_equal(back.original, original)
    assert np.array_equal(back.tags_at(3), wider) and np.array_equal(back.tags_at(4), original)
    assert back.result_at(3) == tl.result_at(3)
    assert back.events[1]["detail"] == {"operation": {"technique": "widen"}}
    assert spectator.saved_timelines(tmp_path) == [path]
