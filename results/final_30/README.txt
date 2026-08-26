Experiment: final_30
Description: canonical 30 seeds, length 2000

final_30: 5 workloads x 3 configs x 30 seeds, length=2000, 32 L1 lines x 8 words, MEM_LATENCY=40 cycles, store-buffer limit=16

Generated: 2026-08-26T10:49:20+00:00
Git revision: 28e83078e3d32ffac358511fff943882d73f595e (dirty: False)
Python: 3.14.6  platform: Linux-7.1.5-arch1-2-x86_64-with-glibc2.44
Library versions: numpy=2.5.2, pandas=3.0.5, scikit_learn=1.9.0, matplotlib=3.11.1, seaborn=0.13.2

Files:
  config.json   - exact experiment specification
  manifest.json - provenance (git rev, versions, host)
  runs.csv      - one row per (seed, workload, config)
  summary.csv   - aggregated means, stds, 95% CI
  speedups.csv  - per-run speedups vs baseline
  table.tex     - LaTeX booktabs table
Figures:
  figures/speedup.png
  figures/hit_rate.png
  figures/cycles.png

Reproduce:  python main.py --config results/final_30/config.json
