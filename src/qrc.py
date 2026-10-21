"""Portable QRC/SQRC implementation of manuscript Section 3.

Matched Base and Raw predictions must use the same pair ordering and score scale.
No labels are read. Estimate tau2 on independent validation queries only.
Singleton-query variance follows the historical implementation's zero convention;
this convention cannot quantify singleton estimation uncertainty.
"""
import numpy as np
import pandas as pd


def query_statistics(base, raw, queries):
    base, raw = np.asarray(base, dtype=float), np.asarray(raw, dtype=float)
    queries = np.asarray(queries)
    if base.ndim != 1 or raw.shape != base.shape or queries.shape != base.shape or not len(base):
        raise ValueError("Provide nonempty aligned one-dimensional arrays")
    if not np.isfinite(base).all() or not np.isfinite(raw).all() or pd.isna(queries).any():
        raise ValueError("Scores must be finite and query IDs must be nonmissing")
    frame = pd.DataFrame({"query": queries, "delta": raw-base})
    stats = frame.groupby("query", sort=False).delta.agg(["size", "mean", "var"])
    stats.columns = ["n", "bias", "variance"]
    stats["variance"] = stats["variance"].fillna(0.0)
    stats["se2"] = stats.variance / stats.n
    return stats


def estimate_tau2(validation_base, validation_raw, validation_queries):
    stats = query_statistics(validation_base, validation_raw, validation_queries)
    if len(stats) < 2:
        raise ValueError("At least two independent validation queries are required")
    return max(float(stats.bias.var(ddof=1) - stats.se2.mean()), 0.0)


def correct(base, raw, queries, tau2=None):
    """Return row-aligned QRC and, with a validation-derived tau2, SQRC scores.

tau2 is not estimated from the supplied test rows. Metadata are query-indexed.
"""
    stats = query_statistics(base, raw, queries)
    raw = np.asarray(raw, dtype=float)
    mapped = stats.reindex(np.asarray(queries))
    output = {"qrc": raw-mapped.bias.to_numpy()}
    if tau2 is not None:
        if not np.isfinite(tau2) or tau2 < 0:
            raise ValueError("tau2 must be finite and nonnegative")
        denominator = tau2 + stats.se2.to_numpy()
        stats["weight"] = np.divide(tau2, denominator, out=np.zeros_like(denominator), where=denominator>0)
        output["sqrc"] = raw - stats.reindex(np.asarray(queries)).weight.to_numpy()*mapped.bias.to_numpy()
    output["statistics"] = stats
    return output
