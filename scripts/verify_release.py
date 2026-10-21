"""Verify immutable release data and saved experiment identities; CPU only."""
from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def records(name):
    path = ROOT / 'data' / name
    with (gzip.open(path, mode='rt', encoding='utf-8-sig', newline='') if path.suffix == '.gz' else path.open(encoding='utf-8-sig', newline='')) as fh:
        return list(csv.DictReader(fh, delimiter='\t' if path.suffix == '.tsv' else ','))


def unique(rows, columns, name):
    keys = [tuple(row[c] for c in columns) for row in rows]
    require(len(keys) == len(set(keys)), f'Duplicate keys: {name}')


def main():
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    report = {'status': 'PASS', 'tables': [], 'checks': {}}
    source_manifest = json.loads((ROOT/'historical/manifest.json').read_text(encoding='utf-8'))
    for item in source_manifest:
        path = ROOT/item['path']
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], f'Historical code hash mismatch: {path.name}')
        compile(path.read_bytes(), str(path), 'exec')
    report['checks']['historical_scripts_hash_and_syntax'] = len(source_manifest)
    for item in manifest:
        path = ROOT/item['path']
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], f'Hash mismatch: {path.name}')
        if item.get('source_sha256'):
            require(hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == item['source_sha256'], 'Decompressed source hash mismatch')
        rows = records(path.name)
        require(len(rows) == item['rows'], f'Row count mismatch: {path.name}')
        require(list(rows[0]) == item['columns'], f'Header mismatch: {path.name}')
        require(all(None not in row and all(v is not None for v in row.values()) for row in rows), f'Ragged rows: {path.name}')
        report['tables'].append({'file': item['path'], 'rows': len(rows), 'sha256': item['sha256']})

    metrics = records('seed_metrics.csv')
    unique(metrics, ['setting', 'seed'], 'seed metrics')
    for setting in {r['setting'] for r in metrics}:
        require({int(r['seed']) for r in metrics if r['setting'] == setting} == set(range(25)), f'Seed coverage: {setting}')

    pairs = records('drug_pair_scores.csv.gz')
    unique(pairs, ['seed', 'drug', 'miRNA'], 'PRISM pairs')
    require(Counter(r['seed'] for r in pairs) == {str(s):7170 for s in range(25)}, 'PRISM seed/pair coverage')
    errors = {'delta': 0., 'qrc': 0., 'sqrc': 0.}
    query_constants = {}
    labels = Counter()
    for row in pairs:
        labels[row['label']] += 1
        for field in ['base_score','raw_score','delta','query_bias','qrc_score','sqrc_weight','sqrc_score']:
            require(math.isfinite(float(row[field])), f'Nonfinite PRISM score: {field}')
        raw, base, bias, weight = (float(row[k]) for k in ['raw_score','base_score','query_bias','sqrc_weight'])
        require(0 <= weight <= 1, 'Invalid SQRC weight')
        errors['delta'] = max(errors['delta'], abs(float(row['delta']) - (raw-base)))
        errors['qrc'] = max(errors['qrc'], abs(float(row['qrc_score']) - (raw-bias)))
        errors['sqrc'] = max(errors['sqrc'], abs(float(row['sqrc_score']) - (raw-weight*bias)))
        key = row['seed'], row['drug']
        if key in query_constants:
            require(query_constants[key] == (bias, weight), 'Within-query shift/weight not constant')
        else:
            query_constants[key] = bias, weight
    require(max(errors.values()) < 1e-10, 'Saved score identity mismatch')
    require(labels == {'0':171803,'1':7447}, 'PRISM label counts')
    report['checks']['prism_score_identity_max_abs_errors'] = errors
    report['checks']['prism_label_counts'] = dict(labels)

    stress = records('sqrc_stress_800_raw.csv')
    unique(stress, ['seed','sigma','n_candidates'], 'stress')
    grid = Counter((float(r['sigma']), int(r['n_candidates'])) for r in stress)
    require(len(grid) == 32 and set(grid.values()) == {25} and len(stress) == 800, 'Stress condition coverage')
    for condition in grid:
        require({int(r['seed']) for r in stress if (float(r['sigma']),int(r['n_candidates'])) == condition} == set(range(25)), 'Stress seed coverage')
    wins = sum(float(r['sqrc_bias_mse']) < float(r['qrc_bias_mse']) for r in stress)
    require(wins == 563, f'Stress wins: {wins}')
    report['checks']['sqrc_lower_mse'] = {'wins': wins, 'cases':len(stress)}
    difficult = [r for r in stress if float(r['sigma']) == 2 and int(r['n_candidates']) == 5]
    report['checks']['sigma2_n5_means'] = {k:sum(float(r[k]) for r in difficult)/25 for k in ['qrc_bias_mse','sqrc_bias_mse','mean_weight']}
    for name, cols in [
        ('davis_boundary_25seeds.csv',['seed','method']),
        ('counterfactual_25seeds.csv',['mode','seed','repeat','condition']),
        ('candidate_stability_25seeds.csv',['mode','seed','fraction','repeat']),
        ('fig3a_cellpainting_seedwise.tsv',['split','seed','method']),
        ('fig3b_crossbackbone_seedwise.tsv',['backbone','seed','method']),
        ('fig3c_prism_seedwise.tsv',['setting','seed','method']),
        ('fig3d_prism_shift_seedwise.tsv',['setting','seed']),
        ('topk_per_seed.csv',['task','setting','seed','method','budget_fraction']),
        ('cross_query_per_seed.csv',['task','setting','seed','method']),
        ('beta_gamma_recovery.csv',['seed','alpha','beta','gamma']),
    ]:
        unique(records(name), cols, name)
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/VERIFICATION.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"PASS: {len(manifest)} data tables; PRISM score identities; SQRC {wins}/800.")


if __name__ == '__main__':
    main()
