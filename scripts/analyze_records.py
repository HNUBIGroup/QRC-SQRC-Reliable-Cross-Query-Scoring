"""Summarize archived results and independently evaluate the PRISM pair table."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from qrc import correct


def main():
    output = ROOT/'results'
    output.mkdir(exist_ok=True)
    primary = pd.read_csv(ROOT/'data/seed_metrics.csv')
    columns = [c for c in primary if c.endswith(('auroc','aupr'))]
    primary.groupby('setting')[columns].agg(['mean','std']).to_csv(output/'cellpainting_summary.csv')
    davis = pd.read_csv(ROOT/'data/davis_boundary_25seeds.csv')
    davis.groupby('method')[['pooled_auroc','pooled_aupr','drug_macro_auroc','drug_macro_aupr']].agg(['mean','std']).to_csv(output/'davis_summary.csv')
    pairs = pd.read_csv(ROOT/'data/drug_pair_scores.csv.gz')
    expected = pd.read_csv(ROOT/'data/fig3c_prism_seedwise.tsv',sep='\t')
    rows = []
    max_score_error = 0.
    max_macro_difference = 0.
    max_archived_metric_error = 0.
    max_within_query_difference_error = 0.
    for seed, frame in pairs.groupby('seed'):
        prediction = correct(frame.base_score, frame.raw_score, frame.drug)['qrc']
        max_score_error = max(max_score_error,float(np.max(np.abs(prediction-frame.qrc_score.to_numpy()))))
        for _, group in frame.groupby('drug'):
            raw = group.raw_score.to_numpy()
            for method in ['qrc','sqrc']:
                corrected = group[method+'_score'].to_numpy()
                error = np.max(np.abs((raw-raw[0])-(corrected-corrected[0])))
                max_within_query_difference_error = max(max_within_query_difference_error,float(error))
        macro_values = {}
        for method in ['base','raw','qrc','sqrc']:
            col = method+'_score'
            eligible = [g for _, g in frame.groupby('drug') if g.label.nunique() == 2]
            row = dict(seed=int(seed), method=method.title() if method in ['base','raw'] else method.upper(),
                       pooled_AUROC=float(roc_auc_score(frame.label,frame[col])),
                       pooled_AUPR=float(average_precision_score(frame.label,frame[col])),
                       macro_AUROC=float(np.mean([roc_auc_score(g.label,g[col]) for g in eligible])),
                       macro_AUPR=float(np.mean([average_precision_score(g.label,g[col]) for g in eligible])))
            rows.append(row)
            macro_values[method] = np.array([row['macro_AUROC'],row['macro_AUPR']])
            match = expected[(expected.seed == seed) & (expected.method.str.lower() == method) & (expected.setting.str.contains('drug',case=False))]
            if len(match) != 1:
                raise ValueError(f'Expected exactly one PRISM drug-cold metric row for {seed}/{method}')
            for metric in ['pooled_AUROC','pooled_AUPR','macro_AUROC','macro_AUPR']:
                max_archived_metric_error = max(max_archived_metric_error,abs(row[metric]-float(match.iloc[0][metric])))
        for method in ['qrc','sqrc']:
            max_macro_difference = max(max_macro_difference,float(np.max(abs(macro_values['raw']-macro_values[method]))))
    if max(max_score_error,max_archived_metric_error,max_macro_difference,max_within_query_difference_error) > 1e-8:
        raise ValueError('Prediction or metric reproduction exceeded tolerance')
    pd.DataFrame(rows).to_csv(output/'prism_drugcold_recomputed.csv',index=False)
    gains = {}
    for setting, frame in primary.groupby('setting'):
        gains[setting] = float((frame.qrc_pooled_auroc-frame.raw_pooled_auroc).mean()*100)
    report = dict(status='PASS',primary_qrc_minus_raw_auroc_percentage_points=gains,
                  prism_qrc_recomputed_score_max_abs_error=max_score_error,
                  prism_archived_metric_max_abs_error=max_archived_metric_error,
                  prism_macro_invariance_max_abs_error=max_macro_difference,
                  prism_within_query_difference_max_abs_error=max_within_query_difference_error,
                  sd_convention='Sample standard deviation (ddof=1)',
                  scope='Archived-result reanalysis; no backbone training or test-derived SQRC tau fitting.')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
