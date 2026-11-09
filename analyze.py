"""Reproduce tables and figures from fixed, run-level evaluation summaries."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
METHODS = {
    "hemopi2_rf1190_author": "HemoPI2 RF",
    "plisson_gbc56": "Plisson GBC",
    "dashboard_bilstm": "Dashboard RNN",
    "aac_logistic_control": "Logistic control",
    "hemopi2_rf450": "Reduced RF control",
}
MAIN = list(METHODS)[:4]
PROTOCOLS = ["published_holdout", "random_holdout", "relatedness_holdout"]
COLORS = ["#267B83", "#C06D34", "#6B61A8", "#909AA5"]
METRICS = ["auroc", "average_precision", "mcc_at_0.5", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "brier", "train_seconds"]

def load_and_validate():
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    for record in provenance:
        p = ROOT / record["file"]
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == record["sha256"]
    d = pd.read_csv(ROOT / "data/run_metrics.csv")
    assert len(d) == 45 and not d.duplicated(["protocol", "seed", "method"]).any()
    assert set(d.method) == set(METHODS)
    assert set(d.protocol) == set(PROTOCOLS)
    assert set(d.seed) == {17, 42, 2026}
    assert np.isfinite(d[METRICS].to_numpy()).all()
    for name in ["auroc", "average_precision", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "brier"]:
        assert d[name].between(0, 1).all()
    assert d["mcc_at_0.5"].between(-1, 1).all()
    # Check confusion-count arithmetic without requiring individual predictions.
    assert (d.tp + d.tn + d.fp + d.fn).eq(d.test_n).all()
    assert np.allclose((d.tp + d.tn) / d.test_n, d.accuracy)
    assert np.allclose(d.tp / (d.tp + d.fn), d.sensitivity)
    assert np.allclose(d.tn / (d.tn + d.fp), d.specificity)
    for (_, _), g in d.groupby(["protocol", "seed"]):
        assert g[["train_n", "validation_n", "test_n"]].nunique().eq(1).all()
    intervals = pd.read_csv(ROOT / "data/paired_auroc_intervals.csv")
    assert (intervals.ci_lower <= intervals.ci_upper).all()
    return d

def main():
    out = ROOT / "results"
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    d = load_and_validate()
    a = d.groupby(["protocol", "method"])[METRICS].agg(["mean", "std"])
    a.columns = [f"{x}_{y}" for x, y in a.columns]
    a.reset_index().to_csv(out / "summary.csv", index=False)
    full = []
    for method in METHODS:
        record = {"method": METHODS[method]}
        for protocol in PROTOCOLS:
            r = a.loc[(protocol, method)]
            record[protocol] = f"{r.auroc_mean:.3f} ± {r.auroc_std:.3f}"
        full.append(record)
    pd.DataFrame(full).to_csv(out / "auroc_table.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=240, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ["auroc", "average_precision", "mcc_at_0.5"], ["AUROC", "Average precision", "MCC at threshold 0.5"]):
        x = np.arange(3)
        for j, (method, color) in enumerate(zip(MAIN, COLORS)):
            y = [a.loc[(p, method), metric + "_mean"] for p in PROTOCOLS]
            e = [a.loc[(p, method), metric + "_std"] for p in PROTOCOLS]
            ax.bar(x + (j - 1.5) * .19, y, yerr=e, width=.18, color=color, capsize=2, label=METHODS[method])
        ax.set_xticks(x, ["Published", "Random", "Relatedness"])
        ax.set(title=title, ylim=(0, 1.03))
        ax.grid(axis="y", alpha=.16);ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    save(fig, "comparison")
    fig, ax = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    x = np.arange(3)
    for j, (method, color) in enumerate(zip(MAIN, COLORS)):
        y = [a.loc[(p, method), "auroc_mean"] for p in PROTOCOLS]
        e = [a.loc[(p, method), "auroc_std"] for p in PROTOCOLS]
        ax.bar(x + (j - 1.5) * .19, y, yerr=e, width=.18, color=color, capsize=3, label=METHODS[method])
    ax.set_xticks(x, ["Published", "Random", "Relatedness"])
    ax.set(ylabel="AUROC", ylim=(0, 1.02))
    ax.grid(axis="y", alpha=.16); ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=9)
    save(fig, "auroc")
    ci = pd.read_csv(ROOT / "data/paired_auroc_intervals.csv")
    q = ci.loc[ci.protocol.eq("relatedness_holdout")].copy()
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    for i, row in enumerate(q.itertuples()):
        ax.plot([row.ci_lower, row.ci_upper], [i, i], color=COLORS[0], lw=2)
        ax.plot(row.auroc_difference, i, 'o', color=COLORS[0], markersize=6)
    ax.axvline(0, color="#777777", linestyle="--", lw=1)
    ax.set_yticks(range(len(q)), ["RF minus " + METHODS[m] for m in q.comparison])
    ax.set(xlabel="Difference in AUROC with fixed 95% interval", title="Paired comparison on the relatedness test set")
    ax.invert_yaxis()
    save(fig, "paired_intervals")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)
    for method, color in zip(MAIN, COLORS):
        r=a.loc[("relatedness_holdout", method)]
        axes[0].scatter(r.train_seconds_mean, r.auroc_mean, color=color, s=50)
        axes[0].annotate(METHODS[method], (r.train_seconds_mean, r.auroc_mean), xytext=(5,4), textcoords="offset points", fontsize=8)
    axes[0].set(xscale="log", xlabel="Recorded local fit and evaluation seconds", ylabel="Mean AUROC", title="Recorded compute cost", ylim=(.55,.95))
    split=pd.read_csv(ROOT/'data/partition_summary.csv')
    for i,p in enumerate(PROTOCOLS):
        values=split.loc[split.protocol.eq(p),'test_fraction_with_related_train_sequence']
        axes[1].bar(i,values.mean(),yerr=values.std(),color=["#C06D34","#909AA5","#267B83"][i],capsize=3)
    axes[1].set_xticks(range(3),['Published','Random','Relatedness'])
    axes[1].set(ylabel="Test fraction with a close training neighbour",ylim=(0,1),title="Partition audit")
    save(fig,"cost_and_partitions")
    export={"method_names":METHODS,"protocols":PROTOCOLS,"summary":a.reset_index().to_dict('records'),"cohort":json.loads((ROOT/'data/cohort_summary.json').read_text()),"paired_intervals":ci.to_dict('records'),"original_replays":pd.read_csv(ROOT/'data/original_replays.csv').fillna('').to_dict('records')}
    (out/'artifact_data.json').write_text(json.dumps(export,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({"status":"passed","run_records":45,"checks":["Input checksums","Complete method protocol seed grid","Finite metric ranges","Confusion-count arithmetic","Equal split sizes within matched comparisons","Ordered fixed interval bounds"],"reproducibility_scope":"Recalculation of aggregate summaries and figures from existing run-level records; fixed interval estimates are displayed, not re-estimated."},indent=2)+'\n')
    print(pd.DataFrame(full).to_string(index=False))
    print("Aggregate analysis checks passed")

if __name__=='__main__':
    main()
