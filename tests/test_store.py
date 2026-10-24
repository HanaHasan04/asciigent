"""The run store is the write-up's source of truth, so its shape is tested."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asciigent.grid import AsciiGrid  # noqa: E402
from asciigent.store import RunStore  # noqa: E402


def test_run_writes_config_steps_and_result(tmp_path):
    store = RunStore(tmp_path)
    run = store.new_run({"phase": "p1", "category": "cat", "prompt_style": "zero_shot"})

    grid, _ = AsciiGrid.from_text("/\\_/\\", rows=4, cols=8, extract=False)
    paths = run.save_grid("turn0", grid)
    run.log("draft", turn=0, parse_clean=True, ink=grid.ink_ratio())
    run.log("critique", turn=0, text="ears are too small")
    result = run.finish(final_score=3.5)
    store.record(result)

    assert (run.dir / "config.json").exists()
    assert (run.dir / "turn0.txt").read_text(encoding="ascii").startswith("/\\_/\\")
    assert (run.dir / paths["png"]).exists()

    steps = [json.loads(l) for l in (run.dir / "steps.jsonl").read_text().splitlines()]
    assert [s["step"] for s in steps] == ["draft", "critique"]

    index = store.load_index()
    assert len(index) == 1
    assert index[0]["final_score"] == 3.5
    assert index[0]["category"] == "cat"


def test_config_records_the_render_style(tmp_path):
    """P2 compares what the model saw, so the render settings are part of the run."""
    store = RunStore(tmp_path)
    run = store.new_run({"phase": "p2"})
    config = json.loads((run.dir / "config.json").read_text())
    assert "render_style" in config
    assert config["render_style"]["font_size"] > 0


def test_run_ids_are_unique_per_config(tmp_path):
    store = RunStore(tmp_path)
    a = store.new_run({"phase": "p1", "category": "cat"})
    b = store.new_run({"phase": "p1", "category": "dog"})
    assert a.run_id != b.run_id
    assert a.run_id.endswith(a.run_id.split("-")[-1])


def test_index_survives_many_runs(tmp_path):
    store = RunStore(tmp_path)
    for i in range(5):
        run = store.new_run({"phase": "p1", "category": f"c{i}"})
        store.record(run.finish(score=i))
    assert [r["score"] for r in store.load_index()] == [0, 1, 2, 3, 4]
