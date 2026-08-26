Experiment: main
Description: Canonical paper dataset: 8 seeds, synthetic streams, numpy MLP

main: 5 workloads x 3 configs x 8 seeds, length=3000, 32 L1 lines x 8 words, MEM_LATENCY=40 cycles, store-buffer limit=16

Generated: 2026-08-23T18:16:11+00:00
Git revision: 205bac1462d05997c5ae02aaeefbd03ca79106be (dirty: True)
Python: 3.12.3  platform: Windows-11-10.0.26200-SP0
Library versions: numpy=2.5.1, pandas=2.3.3, scikit_learn=1.8.0, matplotlib=3.10.7, seaborn=0.13.2

Files:
  config.json   - exact experiment specification
  manifest.json - provenance (git rev, versions, host)
  runs.csv      - one row per (seed, workload, config)
  summary.csv   - aggregated means, stds, 95% CI
  speedups.csv  - per-run speedups vs baseline
  table.tex     - LaTeX booktabs table
Figures:
  figures\speedup.png
  figures\hit_rate.png
  figures\cycles.png

Reproduce:  python main.py --name main
