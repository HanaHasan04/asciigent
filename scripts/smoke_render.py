"""Eyeball check: parse a drawing, render it, and render both scramble controls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asciigent.grid import AsciiGrid  # noqa: E402
from asciigent.render import save_png  # noqa: E402

CAT = r"""```
 /\_/\
( o.o )
 > ^ <
```"""

out = Path(__file__).resolve().parents[1] / "artifacts" / "smoke"
out.mkdir(parents=True, exist_ok=True)

grid, rep = AsciiGrid.from_text(CAT, rows=8, cols=16)
print("parse report:", rep.to_dict())
print("ink %.3f  bbox %s  digest %s" % (grid.ink_ratio(), grid.bbox(), grid.digest()))
print(grid.numbered_text())

save_png(grid, str(out / "cat.png"))
save_png(grid.scrambled("cells", seed=0), str(out / "cat_scramble_cells.png"))
save_png(grid.scrambled("ink", seed=0), str(out / "cat_scramble_ink.png"))
print("wrote 3 PNGs to", out)
