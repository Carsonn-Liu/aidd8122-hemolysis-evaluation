# Evaluation of Published Peptide Hemolysis Classifiers

This AIDD8122 project examines existing classifier evaluation records. It compares predictive performance under common data partitions, describes uncertainty, and audits the extent to which original published scores were recovered.

The repository contains the complete source for the **aggregate-results analysis** used in the accompanying report. Its inputs are 45 run-level records and fixed study-level summaries. It regenerates numerical tables and scientific figures. Predictor training, model weights, peptide sequences and individual prediction records are outside this package and its reproducibility claims.

## Run the analysis

Python 3.12 was used. Create an isolated environment, install `requirements.txt`, then run:

```bash
python analyze.py
```

Outputs appear in `results/`: the full metric summary, formatted AUROC table, editable SVG figures, PNG figures, artifact data, and a validation record. `analyze.py` verifies source checksums, the 45-record comparison grid, matched partition sizes, metric ranges and confusion-count arithmetic before aggregation.

## Inputs

- `run_metrics.csv`: one row per method, partition protocol and random seed.
- `cohort_summary.json`: aggregate cohort and relatedness counts.
- `partition_summary.csv`: partition sizes, class fractions and distance summaries.
- `original_replays.csv`: recorded results from the previously executed original-setting checks.
- `fixed_split_auroc_intervals.csv` and `paired_auroc_intervals.csv`: interval estimates from the existing experiment record. The analysis displays these fixed estimates; it does not recompute their bootstrap procedure.
- `provenance.json`: SHA256 for each aggregate input.

Three seeds describe repeatability within this dataset. They are not independent biological validation cohorts. Mean and standard deviation describe variation among records. Different partition protocols change sample composition and training size, so their performance difference is not a causal estimate of leakage.

## Literature

1. Rathore et al. (2025). Prediction of hemolytic peptides and their hemolytic concentration. Communications Biology 8, 176. https://doi.org/10.1038/s42003-025-07615-w
2. Plisson et al. (2020). Machine learning-guided discovery and design of non-hemolytic peptides. Scientific Reports 10, 16581. https://doi.org/10.1038/s41598-020-73644-6
3. Ansari and White (2023). Serverless Prediction of Peptide Properties with Recurrent Neural Networks. Journal of Chemical Information and Modeling 63, 2546–2553. https://doi.org/10.1021/acs.jcim.2c01317

The project reports method-component reimplementations and score-reproduction differences. It does not label an architecture port or a component evaluation as complete original-paper reproduction.
