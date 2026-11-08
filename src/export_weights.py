"""Export a trained checkpoint to a compact half-precision file for the repository: {"encoder", "state_dict"}."""
import argparse, json, os, torch

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("out"); ap.add_argument("--ckpt", default="best.pt")
a = ap.parse_args()
enc = json.load(open(os.path.join(a.run_dir, "config.json")))["encoder"]
sd = torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu")
sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save({"encoder": enc, "state_dict": sd}, a.out)
print(f"{a.out}: {os.path.getsize(a.out)/1e6:.1f} MB, encoder {enc}")
