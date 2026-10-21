"""Compare rerun stress records with all 24 original archived fields."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('reproduced_csv',type=Path)
args = parser.parse_args()
original = pd.read_csv(ROOT/'data/sqrc_stress_800_raw.csv')
reproduced = pd.read_csv(args.reproduced_csv)
keys = ['seed','sigma','n_candidates']
if len(reproduced) != 800 or reproduced.duplicated(keys).any():
    raise ValueError('Expected 800 unique reproduced stress conditions')
original = original.set_index(keys).sort_index()
reproduced = reproduced.set_index(keys).sort_index()
if not original.index.equals(reproduced.index):
    raise ValueError('Condition grid mismatch')
errors = {}
for col in reproduced:
    if col not in original:
        raise ValueError(f'Unexpected field: {col}')
    a,b = original[col].to_numpy(),reproduced[col].to_numpy()
    if not np.allclose(a,b,atol=1e-10,rtol=1e-10,equal_nan=True):
        raise ValueError(f'Reproduction mismatch: {col}')
    errors[col] = float(np.max(np.abs(a-b)))
report = {'status':'PASS','cases':800,'original_columns_compared':len(reproduced.columns)+len(keys),
          'max_abs_error':max(errors.values()),'per_column_max_abs_error':errors}
(ROOT/'results/STRESS_REPRODUCTION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
