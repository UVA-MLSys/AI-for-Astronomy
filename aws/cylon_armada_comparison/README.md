# Cosmic AI on AWS Lambda compared with Cylon Armada

This folder sets the published Cosmic AI results (Staylor et al., arXiv:2501.06249) beside two
reruns of the same workload made with [Cylon Armada](https://github.com/mstaylor/cylon-armada).
The figures are drawn the way the paper drew them.

- **Published.** The results in `../results` and `../figures`, unchanged.
- **Subexperiment A.** The published design rerun with the same configuration. Workers write
  their results to S3 and a Summarize step collects them after the Map state.
- **Subexperiment B.** The same workload, but workers gather their results along a binomial tree
  of direct connections through Cylon inside the Map state, so the S3 collection step goes away.

Both subexperiments cover the paper's grid: partitions of 25, 50, 75 and 100 MB, data sizes of
1, 2, 4, 6, 8, 10 and 12.6 GB (up to 517 workers), and a batch size sweep from 32 to 512. Each
scaling configuration ran 3 times, as in the paper, and each batch size configuration ran 4 times.
Every error bar and std is the sample standard deviation over the runs.

## Which time to compare

End to end time (execution start to execution end) is the fair comparison between A and B.
Subexperiment B gathers its results inside the Map state, so its Map state time includes result
collection, while Subexperiment A collects its results after the Map state ends. Map state time is
kept because it is the paper's own measure (the paper calls it inference time).

## Headline at 12.6 GB

End to end time in seconds, mean ± std.

| Partition | Workers | Published | A | B | B speedup over A |
|---|---|---|---|---|---|
| 25 MB | 517 | 55.95 ± 1.08 | 55.87 ± 6.09 | 29.61 ± 3.63 | 1.89x |
| 50 MB | 259 | 44.89 ± 1.47 | 37.00 ± 0.59 | 34.67 ± 5.14 | 1.07x |
| 75 MB | 173 | 49.35 ± 3.88 | 37.17 ± 1.64 | 37.31 ± 2.57 | 1.00x |
| 100 MB | 130 | 59.24 ± 0.23 | 40.93 ± 4.56 | 41.45 ± 1.78 | 0.99x |

## Contents

### figures/plots_notebook

The four figures of `../plots.ipynb`, drawn per system with the notebook's fonts, colors, markers,
sizes and `../plot_config.py`. Only the title names the system.

| File | Original in `../figures` |
|---|---|
| `{published,A,B}_inference_time.png` | `inference_time.jpg` (Map state time against data size) |
| `{published,A,B}_throughput.png` | `throughput.jpg` (adjusted throughput) |
| `{published,A,B}_batch_varying_inference_time.png` | `batch_varying_inference_time.jpg` |
| `{published,A,B}_batch_varying_throughput.png` | `batch_varying_throughput.jpg` |

The scaling figures share y limits across the three systems. The batch figures share y limits for
A and B only, because the published sweep used a different configuration (see Caveats).

### figures/paper_fig8_fig9

The paper's Fig. 8 and Fig. 9 (`../figures/cai_inference_time.jpg` and `cai_throughput.jpg`, drawn
by `code/Anomaly Detection/Plots/CosmicAIPlots.ipynb`), redrawn per system with shared y limits.

### figures/comparison

Published, A and B on one figure per metric, one panel per partition, with error bars.

| File | Metric |
|---|---|
| `compare_end_to_end_time.png` | End to end time |
| `compare_map_state_time.png` | Map state time |
| `compare_cpu_time.png` | Per worker CPU time (outliers beyond 1.5 IQR dropped, as `../stats.py` does) |
| `compare_adjusted_throughput.png` | Total bits divided by Map state time |

### tables

`cosmic_ai_paper_tables.xlsx` has one sheet per table plus a Notes sheet, and each table is also a
CSV. Throughput is in billion bits per second.

| File | Content |
|---|---|
| `table6_cost.csv` | Paper Table 6 for each system, plus the billed Lambda cost |
| `scaling_results.csv` | Every partition and data size, all four metrics for each system, and the B speedup over A |
| `headline_12_6_gb.csv` | All four metrics at 12.6 GB |
| `batch_size_sweep.csv` | Batch sizes 32 to 512 for each system |

### results/A and results/B

The six CSVs of `../results` in the same layout and columns, produced by `../stats.py` logic from
each subexperiment's runs: `result_stats.csv`, `result_stats_adjusted.csv`,
`batch_varying_results.csv`, `batch_varying_adjusted.csv`, `state_machine_logs.csv` and
`total_execution_time.csv`.

## Caveats

- Subexperiments A and B ran in a different AWS account and period than the published runs, and A
  and B ran in two accounts at most nine days apart, from the same configuration.
- 11 Subexperiment A runs (1 GB at 50, 75 and 100 MB, and 2 GB at 100 MB) take the Map state time
  from the Lambda logs because their Step Functions history no longer exists. On runs that still
  have their history this reads about 0.13 s longer than the Map state.
- Per worker CPU time depends on the account. Subexperiment A's batch runs took a median 35.9 s in
  the first account and 21.5 s in the second, where all of Subexperiment B ran.
- The paper's batch sweep used 25 MB partitions at 1 GB (41 workers, 3 runs). This sweep used
  100 MB partitions at 1 GB (11 workers, 4 runs), so the published batch values are not directly
  comparable with A and B.
- The published end to end times come from `../results/result_stats_adjusted.csv` (1,216 s summed
  over the 28 configurations), not the hand entered `total_execution_time.csv` (1,239 s).
- The Table 6 estimate uses the paper's formula (requests x CPU time x peak memory x
  $0.0000166667 per GB second). The billed cost prices the configured 10,240 MB for each billed
  duration plus the Initialize and Summarize functions.

## Regenerating

These files are produced in cylon-armada by `target/shared/scripts/results/cosmic_paper_deliverables.py`
and `cosmic_repo_format.py` from the raw run logs.