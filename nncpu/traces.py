"""Memory trace I/O and synthetic "ML/agent" access patterns.

Two ways to get traces into the simulator:

* **Record a real one** from your own code::

      with trace_recorder("my.trace") as rec:
          for i in range(n):
              rec.load(base + i)       # a REAL access you just performed
              rec.store(other + i, i)

* **Use a built-in pattern** that mirrors a real workload class:

      kv_cache_append(n)      -- LLM inference: sequential KV writes + attention gathers
      embedding_lookup(n)     -- sparse embedding table reads
      agent_rag(n)            -- RAG/tool loop: long sequential scans + scattered lookups
      token_stream(n)         -- pure sequential token processing

Trace file format (text, one instruction per line, ``#`` comments):

    LOAD 0x1000
    STORE 0x1a08 42

``LOAD 0x1000``  ->  ``{"type": "LOAD", "address": 0x1000}``
``STORE 0x1a08 42`` -> ``{"type": "STORE", "address": 0x1a08, "value": 42}``
"""

import os
import random
from typing import Iterator, List, Optional

from .workloads import REGION

OP_TYPES = ("LOAD", "STORE", "ADD", "MUL", "DIV")
WORD_BYTES = 8

# -- I/O ----------------------------------------------------------------------


def write_trace(path: str, instructions: List[dict]) -> None:
    """Serialize instruction dicts to the text trace format."""
    lines = []
    for inst in instructions:
        t = inst["type"]
        if t == "LOAD":
            lines.append(f"LOAD {inst['address']:#x}")
        elif t == "STORE":
            lines.append(f"STORE {inst['address']:#x} {inst['value']}")
        else:
            a, b = inst.get("operands", (0, 0))
            lines.append(f"{t} {a} {b}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_trace(path: str, limit: Optional[int] = None) -> List[dict]:
    """Parse a text trace file into instruction dicts."""
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            op = parts[0].upper()
            if op not in OP_TYPES:
                raise ValueError(f"Unknown trace op {op!r} in {path!r}")
            if op == "LOAD":
                inst = {"type": "LOAD", "address": int(parts[1], 0)}
            elif op == "STORE":
                inst = {"type": "STORE", "address": int(parts[1], 0),
                        "value": int(parts[2], 0) if len(parts) > 2 else 0}
            else:
                inst = {"type": op,
                        "operands": [int(parts[1], 0),
                                     int(parts[2], 0) if len(parts) > 2 else 0]}
            out.append(inst)
            if limit and len(out) >= limit:
                break
    return out


def read_champsim(
    path: str, limit: Optional[int] = None, word_bytes: int = WORD_BYTES
) -> List[dict]:
    """Read the classic ChampSim trace format (one line per access:
    ``0 <addr> <pc>`` = load, ``1 <addr> <pc>`` = store)."""
    if word_bytes <= 0:
        raise ValueError("word_bytes must be > 0")
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            parts = raw.split()
            if len(parts) < 2:
                continue
            try:
                is_write = int(parts[0]) != 0
                # ChampSim-style addresses are byte addresses; the simulator's
                # cache geometry is expressed in words.
                addr = int(parts[1], 16) // word_bytes
            except ValueError:
                continue
            inst = {"type": "STORE" if is_write else "LOAD", "address": addr}
            if len(parts) >= 3:
                try:
                    inst["pc"] = int(parts[2], 16)
                except ValueError:
                    pass
            if is_write:
                inst["value"] = 0
            out.append(inst)
            if limit and len(out) >= limit:
                break
    return out


class trace_recorder:
    """Captures the addresses your code touches and writes them to a trace.

    Usage::

        with trace_recorder("demo.trace") as rec:
            rec.load(0x1000)
            rec.store(0x1008, 42)
    """

    def __init__(self, path: str):
        self.path = path
        self._buffer: List[dict] = []

    def load(self, address: int) -> None:
        self._buffer.append({"type": "LOAD", "address": int(address)})

    def store(self, address: int, value: int) -> None:
        self._buffer.append({"type": "STORE", "address": int(address),
                             "value": int(value)})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        write_trace(self.path, self._buffer)
        return False


# -- workload patterns ---------------------------------------------------------


def kv_cache_append(n: int = 2000, base: int = 0x4000, seq_window: int = 512,
                    vector_words: int = 16) -> Iterator[dict]:
    """LLM-style KV cache traffic.

    Every --token-- the model writes the key/value vector for the current
    position (sequential append over a bounded buffer), and periodically
    reads back a past position (attention gather).  Mostly sequential with
    scattered far lookups -- a realistic mixed stream.
    """
    window_words = seq_window * vector_words
    slot = 0
    for i in range(n):
        if i % 3 != 0:  # append a KV vector
            pos = (slot % window_words)
            yield {"type": "STORE", "address": base + pos, "value": i}
            slot += vector_words
        if i % 8 == 0:  # attention gather over a past position
            past = (i * 7) % window_words
            yield {"type": "LOAD", "address": base + past}


def embedding_lookup(n: int = 2000, base: int = 0x8000,
                     table_words: int = 4096, vector_words: int = 16,
                     seed: int = 0x5eed) -> Iterator[dict]:
    """Sparse embedding reads (recommendation / retrieval): per lookup, read
    `vector_words` contiguous words, then jump to a pseudo-random row."""
    state = seed
    for _ in range(n // vector_words):
        state = (1103515245 * state + 12345) % (2 ** 31)
        row_base = base + (state % (table_words - vector_words))
        for w in range(vector_words):
            yield {"type": "LOAD", "address": row_base + w}


def agent_rag(n: int = 2000, base: int = 0xC000, chunk: int = 400,
              region: int = 4096, seed: int = 0xaced) -> Iterator[dict]:
    """RAG / tool-calling agent loop: long sequential context scans, then a
    burst of scattered retrieval lookups, repeated -- the phased shape."""
    state = seed
    i = 0
    while i < n:
        # sequential scan of a context chunk
        for _ in range(chunk):
            if i >= n:
                return
            yield {"type": "LOAD", "address": base + i % (chunk * 8)}
            i += 1
        # scattered lookup burst
        for _ in range(chunk // 16):
            if i >= n:
                return
            state = (1103515245 * state + 12345) % (2 ** 31)
            yield {"type": "LOAD", "address": base + (state % region)}
            i += 1


def token_stream(n: int = 2000, base: int = 0x10000) -> Iterator[dict]:
    """Pure sequential token processing (a trivial control case)."""
    for i in range(n):
        yield {"type": "LOAD", "address": base + i}


TRACE_PATTERNS = {
    "kv_cache_append": kv_cache_append,
    "embedding_lookup": embedding_lookup,
    "agent_rag": agent_rag,
    "token_stream": token_stream,
}


def build_trace_workloads(length: int = 2000, seed: Optional[int] = None) -> dict:
    """Build paired trace workloads, optionally varying stochastic patterns."""
    embedding_seed = 0x5EED if seed is None else seed
    agent_seed = 0xACED if seed is None else seed
    return {
        "kv_cache_append": list(kv_cache_append(length)),
        "embedding_lookup": list(embedding_lookup(length, seed=embedding_seed)),
        "agent_rag": list(agent_rag(length, seed=agent_seed)),
        "token_stream": list(token_stream(length)),
    }


# -- run a trace battery and store an experiment -------------------------------


def run_trace_battery(root: str = "results",
                      name: str = "exp_traces",
                      seeds: int = 5,
                      length: int = 2000,
                      workloads: Optional[dict] = None,
                      configs=None) -> dict:
    """Run every trace pattern under the given configs for `seeds` seeds and
    persist in the same schema as experiment.py (summary.csv + runs.csv), so
    the dashboard picks them up."""
    import json
    import time

    import pandas as pd

    from .benchmark import run_workload, summarize
    from .experiment import aggregate

    vary_workloads = workloads is None
    if workloads is None:
        workloads = build_trace_workloads(length, seed=0)
    if configs is None:
        from .benchmark import CONFIGS
        configs = CONFIGS

    rows = []
    for run in range(seeds):
        run_workloads = build_trace_workloads(length, seed=run) if vary_workloads else workloads
        for wname, insts in run_workloads.items():
            for cfg in configs:
                rep = run_workload(insts, cfg,
                                   nn_kwargs={"random_state": run})
                row = summarize(rep)
                row["run"] = run
                row["seed"] = run
                row["workload"] = wname
                row["config"] = cfg
                rows.append(row)
    runs_df = pd.DataFrame(rows)
    if "baseline" in runs_df.config.values:
        paired_baseline = (
            runs_df[runs_df.config == "baseline"]
            [["run", "seed", "workload", "cycles"]]
            .rename(columns={"cycles": "base_cycles"})
        )
        runs_df = runs_df.merge(
            paired_baseline,
            on=["run", "seed", "workload"],
            how="left",
            validate="many_to_one",
        )
        runs_df["speedup"] = runs_df["base_cycles"] / runs_df["cycles"]
    summary_df = aggregate(runs_df)

    outdir = os.path.join(root, name)
    os.makedirs(outdir, exist_ok=True)
    runs_df.to_csv(os.path.join(outdir, "runs.csv"), index=False)
    summary_df.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({
            "kind": "trace_battery", "name": name,
            "seeds": seeds, "length": length,
            "patterns": list(workloads),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    return {"runs": runs_df, "summary": summary_df, "outdir": outdir}


if __name__ == "__main__":
    # quick self-test: record a tiny real trace then replay it
    demo = os.path.join(os.path.dirname(__file__), "..", "demo_trace.txt")
    with trace_recorder(demo) as rec:
        for j in range(64):
            rec.store(0x1000 + (j % 16), j)
            rec.load(0x1000 + (j % 16))
    inst = read_trace(demo)[:4]
    print("recorded trace round-trip OK:", inst)
