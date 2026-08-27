#!/usr/bin/env python3
"""Run and verify a resource-bounded speculative-decoding endpoint audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import median


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def request_json(url: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_command(config: dict, policy: dict, port: int) -> list[str]:
    resources = config["resources"]
    command = [
        "nice", "-n", str(resources["nice"]),
        "taskset", "-c", resources["cpu_affinity"],
        str(resolve(config["runtime"]["server"])),
        "-m", str(resolve(config["models"]["target"]["path"])),
        "-t", str(resources["threads"]),
        "-tb", str(resources["threads"]),
        "-c", str(resources["context"]),
        "-b", str(resources["batch"]),
        "-ub", str(resources["batch"]),
        "-np", str(resources["parallel"]),
        "--no-warmup", "--no-webui",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    if policy["draft_max"]:
        command.extend(
            [
                "-md", str(resolve(config["models"]["draft"]["path"])),
                "--spec-type", "draft-simple",
                "--spec-draft-n-max", str(policy["draft_max"]),
                "--spec-draft-p-min", str(policy["p_min"]),
                "--spec-draft-p-split", str(config.get("draft_p_split", 0.0)),
                "-td", str(resources["threads"]),
                "-tbd", str(resources["threads"]),
            ]
        )
    return command


class Server:
    def __init__(self, config: dict, policy: dict, log_path: Path):
        self.config = config
        self.policy = policy
        self.log_path = log_path
        self.port = free_port()
        self.process: subprocess.Popen | None = None
        self.log_stream = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = self.log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(self.config["resources"]["threads"])
        self.process = subprocess.Popen(
            server_command(self.config, self.policy, self.port),
            cwd=REPO,
            env=environment,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited early; see {self.log_path}")
            try:
                if request_json(f"http://127.0.0.1:{self.port}/health", timeout=1).get("status") == "ok":
                    return self
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(0.2)
        raise RuntimeError(f"server startup timed out; see {self.log_path}")

    def completion(self, prompt: str, n_predict: int) -> tuple[dict, float]:
        generation = self.config["generation"]
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": generation["temperature"],
            "seed": generation["seed"],
            "cache_prompt": generation["cache_prompt"],
            "stream": False,
        }
        for name in ("ignore_eos", "top_k", "top_p", "min_p"):
            if name in generation:
                payload[name] = generation[name]
        start = time.monotonic_ns()
        response = request_json(f"http://127.0.0.1:{self.port}/completion", payload)
        wall_ms = (time.monotonic_ns() - start) / 1e6
        return response, wall_ms

    def __exit__(self, exc_type, exc, traceback):
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        if self.log_stream is not None:
            self.log_stream.close()


def manifest(config_path: Path, config: dict, prompts_path: Path) -> dict:
    binary = resolve(config["runtime"]["server"])
    return {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(config_path.relative_to(REPO)),
        "config_sha256": sha256(config_path),
        "prompts_path": str(prompts_path.relative_to(REPO)),
        "prompts_sha256": sha256(prompts_path),
        "harness_sha256": sha256(Path(__file__).resolve()),
        "runtime_commit": config["runtime"]["commit"],
        "runtime_binary_sha256": sha256(binary),
        "model_sha256": {
            name: sha256(resolve(model["path"])) for name, model in config["models"].items()
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
    }


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_experiment(config_path: Path, output: Path, phase: str, repeats: int) -> None:
    config = load_json(config_path)
    prompts_path = resolve(config["prompts"])
    prompt_artifact = load_json(prompts_path)
    prompts = [item for item in prompt_artifact["prompts"] if item["split"] == phase]
    if not prompts:
        raise SystemExit(f"no prompts for phase {phase!r}")

    for model in config["models"].values():
        observed = sha256(resolve(model["path"]))
        if observed != model["sha256"]:
            raise SystemExit(f"model hash mismatch for {model['path']}: {observed}")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        data = manifest(config_path, config, prompts_path)
        data.update(
            {
                "phase": phase,
                "repeats": repeats,
                "expected_n_predict": config["generation"]["n_predict"],
                "require_byte_identical_output": config.get("require_byte_identical_output", True),
            }
        )
        manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        existing_manifest = load_json(manifest_path)
        if existing_manifest["phase"] != phase or existing_manifest["repeats"] != repeats:
            raise SystemExit("existing manifest does not match requested phase/repeats")

    runs_path = output / "runs.jsonl"
    existing = load_runs(runs_path)
    completed = {(row["repeat"], row["policy"], row["prompt_id"]) for row in existing}
    policies = config["policies"]
    logs = REPO / ".research" / "specdec_logs" / output.name

    with runs_path.open("a", encoding="utf-8") as stream:
        for repeat in range(repeats):
            ordered_policies = policies.copy()
            random.Random(20260827 + repeat).shuffle(ordered_policies)
            for policy_order, policy in enumerate(ordered_policies):
                pending = [p for p in prompts if (repeat, policy["name"], p["id"]) not in completed]
                if not pending:
                    continue
                log_path = logs / f"r{repeat:02d}-{policy_order:02d}-{policy['name']}.log"
                print(f"repeat={repeat} policy={policy['name']} pending={len(pending)}", flush=True)
                with Server(config, policy, log_path) as server:
                    server.completion("Continue this sequence briefly: one, two, three,", 16)
                    random.Random(2026082700 + repeat).shuffle(pending)
                    for prompt_order, prompt in enumerate(pending):
                        response, wall_ms = server.completion(
                            prompt["prompt"], config["generation"]["n_predict"]
                        )
                        timings = response["timings"]
                        if timings["predicted_n"] <= 0:
                            raise RuntimeError(f"empty completion for {prompt['id']}")
                        content = response["content"]
                        row = {
                            "schema_version": 1,
                            "phase": phase,
                            "repeat": repeat,
                            "policy_order": policy_order,
                            "prompt_order": prompt_order,
                            "policy": policy["name"],
                            "draft_max": policy["draft_max"],
                            "p_min": policy["p_min"],
                            "prompt_id": prompt["id"],
                            "dataset": prompt["dataset"],
                            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                            "content_bytes": len(content.encode("utf-8")),
                            "stop_type": response["stop_type"],
                            "predicted_n": timings["predicted_n"],
                            "predicted_ms": timings["predicted_ms"],
                            "predicted_per_second": timings["predicted_per_second"],
                            "prompt_n": timings["prompt_n"],
                            "prompt_ms": timings["prompt_ms"],
                            "request_wall_ms": wall_ms,
                            "draft_n": timings.get("draft_n", 0),
                            "draft_n_accepted": timings.get("draft_n_accepted", 0),
                        }
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        completed.add((repeat, policy["name"], prompt["id"]))
                        print(
                            f"  {prompt['id']}: {row['predicted_per_second']:.2f} t/s, "
                            f"draft={row['draft_n_accepted']}/{row['draft_n']}",
                            flush=True,
                        )


def compute_summary(
    runs: list[dict], expected_repeats: int | None = None, expected_n_predict: int | None = None
) -> dict:
    if not runs:
        raise ValueError("no runs")
    keys = [(row["repeat"], row["policy"], row["prompt_id"]) for row in runs]
    duplicates = len(keys) - len(set(keys))
    policies = sorted({row["policy"] for row in runs})
    prompts = sorted({row["prompt_id"] for row in runs})
    phases = sorted({row["phase"] for row in runs})
    if len(phases) != 1:
        raise ValueError(f"mixed phases: {phases}")

    hashes_by_prompt = defaultdict(set)
    for row in runs:
        hashes_by_prompt[row["prompt_id"]].add(row["content_sha256"])
    output_mismatches = sorted(prompt for prompt, hashes in hashes_by_prompt.items() if len(hashes) != 1)
    token_count_mismatches = []
    if expected_n_predict is not None:
        token_count_mismatches = [
            {
                "repeat": row["repeat"],
                "policy": row["policy"],
                "prompt_id": row["prompt_id"],
                "predicted_n": row["predicted_n"],
            }
            for row in runs
            if row["predicted_n"] != expected_n_predict
        ]

    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["policy"], row["prompt_id"])].append(row)
    prompt_medians = {}
    for key, rows in grouped.items():
        prompt_medians[key] = {
            "tps": median(row["predicted_per_second"] for row in rows),
            "ms": median(row["predicted_ms"] for row in rows),
            "wall_ms": median(row["request_wall_ms"] for row in rows),
        }

    baseline = "target_only"
    if baseline not in policies:
        raise ValueError("target_only baseline is missing")
    policy_summary = {}
    for policy in policies:
        policy_rows = [row for row in runs if row["policy"] == policy]
        ratios = [
            prompt_medians[(policy, prompt)]["tps"] / prompt_medians[(baseline, prompt)]["tps"]
            for prompt in prompts
            if (policy, prompt) in prompt_medians and (baseline, prompt) in prompt_medians
        ]
        drafted = sum(row["draft_n"] for row in policy_rows)
        accepted = sum(row["draft_n_accepted"] for row in policy_rows)
        policy_summary[policy] = {
            "prompt_units": len(ratios),
            "geomean_tps": geometric_mean(
                [prompt_medians[(policy, prompt)]["tps"] for prompt in prompts]
            ),
            "geomean_speedup_vs_target": geometric_mean(ratios),
            "aggregate_draft_n": drafted,
            "aggregate_draft_n_accepted": accepted,
            "aggregate_acceptance": accepted / drafted if drafted else None,
        }

    reversals = []
    speculative = [p for p in policies if p != baseline]
    for left_index, left in enumerate(speculative):
        for right in speculative[left_index + 1 :]:
            left_acc = policy_summary[left]["aggregate_acceptance"]
            right_acc = policy_summary[right]["aggregate_acceptance"]
            if left_acc is None or right_acc is None or abs(left_acc - right_acc) < 0.01:
                continue
            high, low = (left, right) if left_acc > right_acc else (right, left)
            ratios = [
                prompt_medians[(high, prompt)]["tps"] / prompt_medians[(low, prompt)]["tps"]
                for prompt in prompts
            ]
            direct = geometric_mean(ratios)
            if direct < 1:
                wins = sum(ratio > 1.000001 for ratio in ratios)
                losses = sum(ratio < 0.999999 for ratio in ratios)
                reversals.append(
                    {
                        "higher_acceptance_policy": high,
                        "lower_acceptance_policy": low,
                        "acceptance_difference": policy_summary[high]["aggregate_acceptance"]
                        - policy_summary[low]["aggregate_acceptance"],
                        "geomean_tps_ratio": direct,
                        "wins": wins,
                        "losses": losses,
                        "ties": len(ratios) - wins - losses,
                        "exact_two_sided_sign_p": exact_sign_p(wins, losses),
                    }
                )

    repeats_seen = sorted({row["repeat"] for row in runs})
    expected = None
    if expected_repeats is not None:
        expected = len(policies) * len(prompts) * expected_repeats
    return {
        "schema_version": 1,
        "phase": phases[0],
        "raw_runs": len(runs),
        "expected_runs": expected,
        "complete": expected is None or (len(runs) == expected and duplicates == 0),
        "duplicates": duplicates,
        "repeats_seen": repeats_seen,
        "prompt_units": len(prompts),
        "policies": policy_summary,
        "output_mismatches": output_mismatches,
        "token_count_mismatches": token_count_mismatches,
        "proxy_reversals": sorted(reversals, key=lambda item: item["geomean_tps_ratio"]),
    }


def write_summary(output: Path) -> dict:
    runs = load_runs(output / "runs.jsonl")
    expected_repeats = load_json(output / "manifest.json")["repeats"]
    manifest_data = load_json(output / "manifest.json")
    summary = compute_summary(runs, expected_repeats, manifest_data.get("expected_n_predict"))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["policy", "prompt_units", "geomean_tps", "geomean_speedup_vs_target", "draft_n", "accepted_n", "acceptance"]
        )
        for policy, values in sorted(summary["policies"].items()):
            writer.writerow(
                [
                    policy,
                    values["prompt_units"],
                    values["geomean_tps"],
                    values["geomean_speedup_vs_target"],
                    values["aggregate_draft_n"],
                    values["aggregate_draft_n_accepted"],
                    values["aggregate_acceptance"],
                ]
            )
    return summary


def verify(output: Path) -> None:
    saved = load_json(output / "summary.json")
    manifest_data = load_json(output / "manifest.json")
    config_path = REPO / manifest_data["config_path"]
    prompts_path = REPO / manifest_data["prompts_path"]
    checks = {
        "config": (sha256(config_path), manifest_data["config_sha256"]),
        "prompts": (sha256(prompts_path), manifest_data["prompts_sha256"]),
        "harness": (sha256(Path(__file__).resolve()), manifest_data["harness_sha256"]),
    }
    config = load_json(config_path)
    checks["runtime"] = (
        sha256(resolve(config["runtime"]["server"])),
        manifest_data["runtime_binary_sha256"],
    )
    for name, model in config["models"].items():
        checks[f"model:{name}"] = (sha256(resolve(model["path"])), manifest_data["model_sha256"][name])
    failed_hashes = [name for name, pair in checks.items() if pair[0] != pair[1]]
    reconstructed = compute_summary(
        load_runs(output / "runs.jsonl"),
        manifest_data["repeats"],
        manifest_data.get("expected_n_predict"),
    )
    if failed_hashes:
        raise SystemExit(f"hash verification failed: {failed_hashes}")
    if reconstructed != saved:
        raise SystemExit("saved summary does not match raw-run reconstruction")
    if not saved["complete"]:
        raise SystemExit("run matrix is incomplete")
    if saved["token_count_mismatches"]:
        raise SystemExit(f"fixed-token invariant failed: {saved['token_count_mismatches']}")
    if manifest_data.get("require_byte_identical_output", True) and saved["output_mismatches"]:
        raise SystemExit(f"greedy output mismatches: {saved['output_mismatches']}")
    print(
        f"verified {saved['raw_runs']} runs, {saved['prompt_units']} prompt units, "
        f"{len(saved['proxy_reversals'])} proxy reversals"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, default=REPO / "benchmarks/specdec_config.json")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--phase", choices=["development", "holdout"], required=True)
    run_parser.add_argument("--repeats", type=int, required=True)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "run":
        run_experiment(args.config.resolve(), args.output.resolve(), args.phase, args.repeats)
        summary = write_summary(args.output.resolve())
        print(json.dumps(summary, indent=2))
    elif args.command == "summarize":
        print(json.dumps(write_summary(args.output.resolve()), indent=2))
    else:
        verify(args.output.resolve())


if __name__ == "__main__":
    main()
