"""Validate a submission CSV: decodable RLEs of size 2048x2048, unique ids, no overlapping masks per image."""
import sys, numpy as np, pandas as pd
from pycocotools import mask as mutils

df = pd.read_csv(sys.argv[1])
assert list(df.columns) == ["filament_id", "segmentation_rle"], df.columns
assert df.filament_id.is_unique, "duplicate filament ids"
assert df.segmentation_rle.notna().all() and not df.segmentation_rle.str.contains("[\"']").any()
df["stem"] = df.filament_id.str.rsplit("_", n=1).str[0]
areas = []
for stem, g in df.groupby("stem"):
    acc = np.zeros((2048, 2048), np.uint8)
    for r in g.segmentation_rle:
        m = mutils.decode({"size": [2048, 2048], "counts": r.encode("ascii")})
        assert m.shape == (2048, 2048); areas.append(int(m.sum())); acc += m
    assert acc.max() <= 1, f"overlap in {stem}"
print(f"OK: {len(df)} rows, {df.stem.nunique()} images, area min/median/max {min(areas)}/{int(np.median(areas))}/{max(areas)}")
