# ChampSim adapter

This directory contains a native ChampSim implementation of the matched
regularity experiment. `nncpu_regularity_stride` is an IP-indexed stride
prefetcher derived from ChampSim's `ip_stride`; the same module runs either
with the 16-delta/35%-support admission gate or with admission disabled. This
keeps predictor state, lookahead, cache placement, and implementation matched.

The adapter targets the official ChampSim `develop` runtime-module interface,
validated initially against commit
`e6530c7293a4d93857f634447554281a3e582516`.

To build from an official ChampSim checkout without copying files into it:

```bash
git checkout e6530c7293a4d93857f634447554281a3e582516
git submodule update --init vcpkg
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install
mkdir -p .csconfig/external
make -j$(nproc) EXTERNAL_MODULE_DIR=/path/to/nncpu/champsim
```

The explicit directory creation works around the external-module dependency
directory not being materialized by the Makefile at the pinned commit.

The runtime-module build produces one `bin/champsim`; select the baseline,
raw control, or gate with `--config` and the corresponding JSON in `config/`.
Fetch the fixed public trace set with:

```bash
/path/to/nncpu/champsim/fetch_dpc3.sh /path/to/traces 4
```

The downloader uses atomic partial files and validates the XZ container. Set
`NNCPU_FULL_TRACE_VERIFY=1` to decompress every complete trace and verify its
integrity checksum before simulation; this is much slower than the default.
`campaign.py` runs all three configurations with identical warm-up and ROI
lengths, captures raw JSON and logs, hashes every input, and reports paired
per-trace contrasts:

```bash
python /path/to/nncpu/champsim/campaign.py \
  --champsim /path/to/ChampSim \
  --trace-dir /path/to/traces \
  --output /path/to/results \
  --warmup 50000000 --simulation 200000000 \
  --jobs 4 \
  $(sed 's/#.*//' /path/to/nncpu/champsim/dpc3_trace_set.txt)
```

The checked-in trace set contains the highest-weight SimPoint from each of
eleven distinct SPEC CPU2017 programs, selected from the official DPC-3
weights archive. This external, deterministic rule avoids choosing traces by
compressed size or observed gate performance. Do not interpret an IPC
difference from a single trace as a population-level result.

The module trains throughout warm-up, but its custom decision, suppression,
and issue counters cover only the measured region of interest. ChampSim's JSON
statistics are also ROI-only.

The C++ adapter files retain the Apache-2.0 notice required by the ChampSim
source they derive from. The rest of nncpu remains MIT licensed.
