# add to check.py
import json
meta = json.load(open(r"C:\Users\Admin\Music\sourav\cdac-download\hqnn live sk\hqnn live pk\models\delhi\multivariate\metadata.json"))
feats = meta["features"]["mv"]["TMP2m"]
for i, f in enumerate(feats):
    print(f"{i:3d}  {f}")