Experiment: main_10x4000
Description: background_run_10seeds_4000_numpy

main_10x4000: 5 workloads x 3 configs x 10 seeds, length=4000, 32 L1 lines x 8 words, MEM_LATENCY=40 cycles, store-buffer limit=16

Generated: 2026-08-23T18:50:08+00:00
Git revision: 3cc5f615a1ae93816d8f351c502223bb7a04eef2 (dirty: True)
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

Reproduce:  python main.py --name main_10x4000
