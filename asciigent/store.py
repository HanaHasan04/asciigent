"""Run records and artifacts.

Four people running overlapping sweeps will produce results nobody can
reassemble at write-up time unless every run writes the same shape.  A run is:

    runs/<run_id>/
        config.json      what was asked for
        steps.jsonl      one line per turn (draft, critique, revision, judge)
        turn0.txt/.png   the drawing at each turn, as text and as rendered
        result.json      final metrics

plus one summary line appended to ``runs/index.jsonl`` so a whole sweep can be
loaded into a dataframe without walking the tree.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

from .grid import AsciiGrid
from .render import RenderStyle, save_png


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, AsciiGrid):
        return obj.to_text()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class Run:
    """One experimental unit: one prompt taken through to a final drawing."""

    def __init__(self, root: Path, run_id: str, config: dict, style: RenderStyle):
        self.dir = root / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.config = config
        self.style = style
        self.started = time.time()
        self._steps_path = self.dir / "steps.jsonl"
        self._write(self.dir / "config.json", config | {"render_style": style.to_dict()})

    def _write(self, path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8"
        )

    def log(self, step: str, **fields: Any) -> None:
        """Append one turn to the run's log."""
        record = {"t": round(time.time() - self.started, 3), "step": step, **fields}
        with self._steps_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_jsonable) + "\n")

    def save_grid(self, name: str, grid: AsciiGrid, png: bool = True) -> dict:
        """Persist a drawing as text and (by default) as the rendered bitmap."""
        txt = self.dir / f"{name}.txt"
        txt.write_text(grid.to_text(), encoding="ascii")
        paths = {"text": txt.name}
        if png:
            save_png(grid, str(self.dir / f"{name}.png"), self.style)
            paths["png"] = f"{name}.png"
        return paths

    def finish(self, **metrics: Any) -> dict:
        result = {
            "run_id": self.run_id,
            "elapsed_s": round(time.time() - self.started, 3),
            **self.config,
            **metrics,
        }
        self._write(self.dir / "result.json", result)
        return result


class RunStore:
    """Creates runs and keeps the flat index."""

    def __init__(self, root: str | Path = "runs", style: RenderStyle | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"
        self.style = style or RenderStyle()

    def new_run(self, config: dict) -> Run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        # Config hash keeps ids stable-ish and collision-free within a second.
        digest = sha1(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:6]
        phase = config.get("phase", "run")
        return Run(self.root, f"{stamp}-{phase}-{digest}", config, self.style)

    def record(self, result: dict) -> None:
        with self.index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, default=_jsonable) + "\n")

    def load_index(self) -> list[dict]:
        if not self.index.exists():
            return []
        return [
            json.loads(line)
            for line in self.index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
