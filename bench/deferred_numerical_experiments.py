#!/usr/bin/env python3
"""Collect deferred numerical work without promoting compilation to GPU evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from spec_sheet import split_decls, split_statement
BASE = "bench/tritonbench_g/"
OUTPUT = BASE + "deferred_numerical_experiments"
SUPPLEMENT = BASE + "parallel_retention_attention/NUMERICAL_ASSUMPTIONS.json"
RULE_STATES = {
    "pending_exact_binding": "pending_exact_binding",
    "pending_numeric_validation": "exact_profile_bound_validation_pending",
    "existing_report_optional_replay": "reported_profile_only",
    "shared_model_check": "model_requirement_not_hardware_evidence",
    "not_an_experiment": "refuted_unrestricted_assumption",
}
JOB_TASK_ALIASES = {
    "chunk_retention (signed32 reference repair)": "chunk_retention",
    "chunk_retention_ops (signed32 reference repair)": "chunk_retention_ops",
    "int8_matmul_quantization (maximum revision)": "int8_matmul_quantization",
    "mixed_sparse_attention (maximum revision)": "mixed_sparse_attention",
}
SUPPLEMENT_STATES = {
    "experiment_not_run", "compiled_graph_proof_and_experiment_pending",
    "semantic_refinement_and_experiment_pending", "outside_proved_scope",
}


class Inputs:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.hashes = {}
        self.statements = {}

    def bind(self, name: str, expected: str | None = None) -> bytes:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Missing or external input: {name}")
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError(f"Changed bound input: {name}")
        self.hashes[name] = actual
        return data

    def read(self, name: str, expected: str | None = None):
        return json.loads(self.bind(name, expected))

    def declaration(self, file: str, row: dict):
        if file not in self.statements:
            source = self.bind(file).decode()
            self.statements[file] = {d["name"]: split_statement(d["text"])[0]
                                     for d in split_decls(source)}
        name = row.get("declaration", row.get("name"))
        statement = self.statements[file].get(name)
        if statement is None or hashlib.sha256(statement.encode()).hexdigest() != row["statement_sha256"]:
            raise ValueError(f"Changed numerical declaration: {file}::{name}")


def unique(rows, key):
    result = {}
    for row in rows:
        name = row[key]
        if name in result:
            raise ValueError(f"Duplicate {key}: {name}")
        result[name] = row
    return result


def collect_legacy(root: Path = ROOT):
    inputs = Inputs(root)
    current = inputs.read(BASE + "fp_value_current_status.json")
    rules = inputs.read(BASE + "numeric_experiment_tasks.json")
    original = inputs.read(BASE + "fp_value_native_tasks.json")
    extensions = inputs.read(BASE + "fp_value_extensions.json")
    native_ref = extensions["native_plan"]
    native = inputs.read(native_ref["path"], native_ref["sha256"])
    supplement = inputs.read(SUPPLEMENT)
    tasks = unique(current["tasks"], "task")
    if len(tasks) != 184 or current["summary"]["tasks"] != 184:
        raise ValueError("The current inventory must contain 184 distinct tasks")
    if (original["native_execution_performed"] or native["native_execution"]
            or extensions["native_gpu_execution"]
            or any(row["native_execution_performed"] for row in original["tasks"])
            or any(job["status"] != "native_experiments_not_run" for job in native["jobs"])):
        raise ValueError("New execution evidence needs an explicit review; do not relabel it as deferred")
    original_rows = unique(original["tasks"], "task")
    extension_rows = unique(extensions["tasks"], "task")
    jobs = unique([{**job, "source_job_label": job["task"],
                    "task": JOB_TASK_ALIASES.get(job["task"], job["task"])}
                   for job in native["jobs"]], "task")
    if (set(original_rows) | set(extension_rows) | set(jobs)) - set(tasks):
        raise ValueError("An experiment refers to an unknown task")

    rule_rows = []
    for rule in unique(rules, "id").values():
        if rule["status"] not in RULE_STATES:
            raise ValueError("Unknown numerical rule status: " + rule["id"])
        if set(rule["affected_tasks"]) - set(tasks):
            raise ValueError("Numerical rule refers to an unknown task")
        disposition = RULE_STATES[rule["status"]]
        if disposition == "reported_profile_only" and not rule.get("existing_report"):
            if not rule.get("existing_evidence"):
                raise ValueError("Reported profile lacks an evidence reference: " + rule["id"])
            disposition = "conditional_on_reported_model"
        rule_rows.append({**rule, "queue_status": disposition,
                          "native_whole_kernel_validated": False})

    task_rows = []
    for name, task in sorted(tasks.items()):
        old = original_rows.get(name)
        ext = extension_rows.get(name)
        job = jobs.get(name)
        declarations = []
        obligations = []
        if old:
            declarations.extend(old["declarations"])
            obligations.extend(old["binding_relations"])
        if ext:
            declarations.extend(ext["declarations"])
        for declaration in declarations:
            file = declaration.get("file") or (ext["public_file"] if ext else old["file"])
            inputs.declaration(file, declaration)
        if ext and not job and not old:
            # Newly proved raw-copy tasks are absent from the old 37-job plan.
            # Empty arithmetic callbacks do not establish native memory behavior.
            obligations.append({
                "id": "native_storage_address_and_composition",
                "relation": "Bind the declared raw words, physical byte ABI, masks, native integer addresses, "
                            "allocations, aliases and program ordering; no arithmetic identity is needed.",
            })
        task_rows.append({
            "task": name,
            "current_value_status": task["current_value_status"],
            "queue_status": "native_experiments_not_run" if old or ext or job else
                            "native_value_contract_not_in_this_queue",
            "scope": ext["accepted_scope"] if ext else old["covered_scope"] if old else job["scope"] if job else
                     "Relative-equivalence or historical model scope; consult the existing declaration review.",
            "dtype_profile": ext["dtype_profile"] if ext else old["native_profile"] if old else None,
            "declarations": declarations,
            "binding_relations": obligations,
            "existing_native_job": job,
            "rule_ids": [r["id"] for r in rule_rows if name in r["affected_tasks"]],
            "unclosed_numeric_callbacks": sorted({c for d in declarations
                for c in d.get("unclosed_numeric_callbacks", [])}),
            "native_execution": "NOT_RUN",
            "native_whole_kernel_validated": False,
        })

    if supplement["task"] not in tasks or supplement["native_execution"] != "NOT_RUN":
        raise ValueError("Invalid supplemental experiment scope/status")
    unique(supplement["obligations"], "id")
    for name in supplement["source_files"]:
        inputs.bind(name)
    for item in supplement["obligations"]:
        if not all(item.get(k) for k in ("id", "status", "relation", "cases", "required_evidence")):
            raise ValueError("Incomplete supplemental numerical obligation")
        if item["status"] not in SUPPLEMENT_STATES:
            raise ValueError("Unreviewed supplemental evidence status: " + item["id"])
    counts = Counter(r["queue_status"] for r in rule_rows)
    return {
        "schema": 1,
        "purpose": "Deferred experiment handoff; software proofs and numerical evidence remain separate.",
        "native_execution_in_this_work": "NOT_RUN",
        "summary": {"inventory_tasks": len(tasks), "rule_profiles": len(rule_rows),
                    "rule_statuses": dict(sorted(counts.items())),
                    "tasks_with_native_recipes": sum(r["queue_status"] == "native_experiments_not_run"
                                                     for r in task_rows),
                    "existing_extension_native_jobs": len(jobs),
                    "supplemental_obligations": len(supplement["obligations"])},
        "rules": rule_rows, "tasks": task_rows, "supplements": [supplement],
        "required_artifacts": native["required_artifacts"],
        "policy": [
            "Record exact source, theorem/profile, compiled graph and hardware configurations before experiments.",
            "Keep raw inputs, intermediates, outputs, frames, seeds and mismatches, including NaN and signed zero.",
            "Offline compilation and finite software witnesses are not GPU execution or universal numerical proofs.",
            "Existing reports apply only to their recorded profile; pending exact bindings remain pending.",
            "Refuted unrestricted assumptions require a repaired statement/model; experiments cannot certify them.",
            "Tasks without a value recipe are not assumed free of numerical or hardware obligations.",
        ],
        "input_sha256": dict(sorted(inputs.hashes.items())),
    }


def collect(root: Path = ROOT):
    from exact_numerical_bindings import derive_authority, extend_queue
    return extend_queue(collect_legacy(root), derive_authority(root))


def markdown(data):
    summary = data["summary"]
    lines = ["# Deferred numerical experiments", "",
             "This is a handoff for later experiments. No GPU experiment was run by this work. "
             "The existing software/backend proofs keep their exact scopes.", "",
             f"The inventory covers {summary['inventory_tasks']} tasks: "
             f"{summary['tasks_with_native_recipes']} have concrete native-binding recipes, "
             "and the others retain their relative-equivalence/historical review scope. "
             "An empty callback list is not hardware validation.", "",
             "## Numerical rule profiles", "",
             "| Profile | Disposition | Affected tasks |", "|---|---|---:|"]
    for rule in data["rules"]:
        lines.append(f"| `{rule['id']}` | `{rule['queue_status']}` | {len(rule['affected_tasks'])} |")
    lines += ["", "Five profiles still need exact binding. Existing reports remain limited to their "
              "recorded configuration. The unrestricted fp16 rounding-is-identity assumption is refuted; "
              "it must be repaired, not submitted as a validation target.", "",
              "## New Retention proof dependencies", "",
              "The new DQ and sequential DQ/DK/DV proofs use explicit M/D backend parameters. "
              "They prove source execution, complete outputs and the memory frame. "
              "They do not identify those parameters with native instructions.", "",
              "| Obligation | Status | Relation to establish |", "|---|---|---|"]
    for item in data["supplements"][0]["obligations"]:
        lines.append(f"| `{item['id']}` | `{item['status']}` | {item['relation']} |")
    lines += ["", "## Reproduction and batch handoff", "",
              "Run from the repository root:", "", "```sh",
              "python3 -B bench/deferred_numerical_experiments.py --check", "```", "",
              "The [JSON queue](deferred_numerical_experiments.json) contains all task scopes, "
              "declaration references, binding relations, existing job recipes, corner cases, "
              "report records and source hashes. The [Retention supplement]"
              "(parallel_retention_attention/NUMERICAL_ASSUMPTIONS.json) gives per-obligation captures.", "",
              "Keep failed and not-run cases. A source/compiled graph mismatch requires a corrected "
              "graph or an explicit scoped relation before numerical comparison; do not silently "
              "reassociate arithmetic, replace undefined padding with zero, or substitute "
              "dotAcc for dot-from-zero plus add.", ""]
    exact = data.get("exact_numerical_summary")
    if exact:
        lines += ["## Exact scalar follow-up bindings", "",
            f"{exact['new_scalar_families']} additional scalar families bind "
            f"{exact['scope_bindings']} exact dtype/port instances across "
            f"{exact['declaration_cases']} declaration cases. All remain unvalidated.", "",
            "The fp64 cases bind the frozen sm90 frontend configuration. ReLU binds only "
            "N=1025, block_size=1024, sm90; other specializations remain unregistered. "
            "The queue preserves structural gaps and adds no accepted original programs.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = collect(args.root)
    outputs = {OUTPUT + ".json": json.dumps(data, indent=2, ensure_ascii=False) + "\n",
               OUTPUT + ".md": markdown(data)}
    for name, content in outputs.items():
        path = args.root / name
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit("Stale deferred experiment queue: " + name)
        else:
            path.write_text(content)
    print(json.dumps(data["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
