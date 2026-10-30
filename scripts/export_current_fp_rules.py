#!/usr/bin/env python3
"""Export the typed current FP rule view without altering historical evidence.

Normally builds CurrentAdmission with Lake. A workspace build receipt can be
used instead with --receipt and --lean; every recorded source/object/import
hash is checked recursively before and after Lean evaluates the typed view.
No experiment, GPU probe, replay, or historical exporter is invoked.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import fp_rule_inventory as inventory

MODULE = "VeriTile.Triton.Float.CurrentAdmission"
NAMESPACE = "VeriTile.Triton.FP.CurrentAdmission"
TYPED_MODULES = {
    MODULE,
    "VeriTile.Triton.Float.ScalarProbeExact",
    "VeriTile.Triton.Float.MaximumProbeExact",
    "VeriTile.Triton.Float.PrecisionProbeExact",
    "VeriTile.Triton.Float.PrecisionIdentityLaws",
    "VeriTile.Triton.Float.ExponentialProbeExact",
    "VeriTile.Triton.Float.CountConversion",
    "VeriTile.Triton.Float.CountConversionExact",
}
# Reviewed source-to-typed-expression bindings. Changing a probe or its domain
# requires reviewing the corresponding Lean source terms and updating this
# manifest; recording a fresh hash alone must never promote a changed probe.
REVIEWED_SOURCE_SHA256 = {
    "experiments/floating_point/kernels.py":
        "f239cb37af1eab11170c912303a82fa4201c681caa1273db7acb0e0e746aa98e",
    "experiments/floating_point/supplement/kernels.py":
        "4b2646ef86ace235e7d1c27aca653bf56db538e81c10a6919a5075080482ad85",
    "scripts/numerical_domains.py":
        "ab978695945069788f1f218a1c4f9189111ca3308789c086b46c9dd1526ec2e3",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(root):
    for path, expected in REVIEWED_SOURCE_SHA256.items():
        if digest(root / path) != expected:
            raise ValueError(f"reviewed probe/source-domain binding changed: {path}")
    return dict(REVIEWED_SOURCE_SHA256)


def verify_receipts(root, receipt_path):
    """Validate the exact compiled closure; retain explicit baseline boundaries."""
    top = json.loads(receipt_path.read_text())
    libraries = [Path(p) for p in top["lean_path"]]
    checked, baseline = {}, {}

    def resolve(module):
        relative = Path(module.replace(".", "/") + ".olean")
        found = next((p / relative for p in libraries if (p / relative).is_file()), None)
        if found is None:
            raise ValueError(f"missing compiled dependency: {module}")
        return found

    def visit(module, required=False):
        if module in checked or module in baseline:
            return
        obj = resolve(module)
        receipt = obj.with_suffix(".workspace-build.json")
        if not receipt.is_file():
            receipt = obj.resolve().with_suffix(".workspace-build.json")
        if not receipt.is_file():
            if required or module in TYPED_MODULES:
                raise ValueError(f"missing typed-source build receipt: {module}")
            baseline[module] = {"object_sha256": digest(obj), "object": str(obj.resolve())}
            if module.startswith("VeriTile."):
                relative = module.replace(".", "/") + ".lean"
                source = root / relative
                original = subprocess.run(["git", "show", "HEAD:" + relative], cwd=root,
                                          capture_output=True)
                if (original.returncode or not source.is_file()
                        or hashlib.sha256(original.stdout).hexdigest() != digest(source)):
                    raise ValueError(f"unreceipted local source differs from the repository baseline: {module}")
                baseline[module]["source_sha256"] = digest(source)
                baseline[module]["source_status"] = "unchanged repository HEAD; compiled baseline boundary"
            return
        record = json.loads(receipt.read_text())
        source = root / (module.replace(".", "/") + ".lean")
        if (record.get("exit_code") != 0 or record["source"] != source.relative_to(root).as_posix()
                or not source.is_file() or digest(source) != record["source_sha256"]):
            raise ValueError(f"stale/unsuccessful source receipt: {module}")
        recorded_object = next((value for key, value in record["objects"].items()
                                if Path(key).suffix == ".olean"), None)
        if recorded_object != digest(obj):
            raise ValueError(f"compiled object differs from receipt: {module}")
        checked[module] = {
            "source_sha256": record["source_sha256"], "object_sha256": recorded_object,
            "receipt_sha256": digest(receipt),
        }
        for dependency, expected in record["dependency_sha256"].items():
            if digest(resolve(dependency)) != expected:
                raise ValueError(f"stale imported object: {module} → {dependency}; rebuild the dependent module")
            if dependency.startswith("VeriTile."):
                visit(dependency)
            else:
                baseline[dependency] = {"object_sha256": expected,
                                        "object": str(resolve(dependency).resolve())}

    visit(MODULE, required=True)
    if not TYPED_MODULES <= set(checked):
        raise ValueError("typed certificate modules are absent from the checked build closure")
    return {"checked_source_receipts": checked, "baseline_compiled_dependencies": baseline}, libraries


def run(command, root, env=None):
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
    if result.returncode:
        raise ValueError(f"Lean command failed ({result.returncode}):\n{result.stdout}{result.stderr}")
    return result.stdout


def lean_view(root, receipt=None, lean=None):
    verify_sources(root)
    if receipt:
        if not lean:
            raise ValueError("--receipt requires the exact --lean executable")
        build, libraries = verify_receipts(root, receipt)
        env = os.environ.copy()
        env["LEAN_PATH"] = os.pathsep.join(str(p) for p in libraries)
        command = [str(lean), "-s", "32768", "--root=" + str(root)]
    else:
        run(["lake", "build", MODULE], root)
        build, env = {"validation": "fresh lake build", "module": MODULE}, None
        command = ["lake", "env", "lean"]
    # Only a tiny proof/export driver is written, never historical evidence.
    with tempfile.TemporaryDirectory(prefix=".current-fp-view-", dir=root) as work:
        driver = Path(work) / "Export.lean"
        driver.write_text(f"import {MODULE}\n#axiomsClean {NAMESPACE}.StrictSource.certificate\n"
                          f"#axiomsClean {NAMESPACE}.inventory_counts\n"
                          f'#eval IO.println ("CURRENT_FP_JSON=" ++ {NAMESPACE}.exportJson.compress)\n')
        output = run([*command, str(driver)], root, env)
    lines = [line.removeprefix("CURRENT_FP_JSON=") for line in output.splitlines()
             if line.startswith("CURRENT_FP_JSON=")]
    if len(lines) != 1:
        raise ValueError("Lean did not export exactly one typed current view")
    if receipt and verify_receipts(root, receipt)[0] != build:
        raise ValueError("build closure changed during export; retry after builders finish")
    verify_sources(root)
    return json.loads(lines[0]), build


def normal_signature(signature):
    result = dict(signature)
    result["guards"] = sorted(result["guards"], key=lambda g: (g["operand"], g["predicate"]))
    return result


def instance_id(signature):
    signature = normal_signature(signature)
    precision = "/".join(signature["precision"][p] for p in inventory.PRECISIONS)
    guards = ",".join(f"{g['operand']}:{g['predicate']}" for g in signature["guards"]) or "none"
    result = signature["rule_id"] + "::" + precision + "::" + guards
    if signature["syntax_domain"]:
        result += "::" + signature["syntax_domain"]
    return result


def checked_view(root, view):
    """Join every exported row back to the immutable evidence, fail closed."""
    if view["schema_version"] != 1:
        raise ValueError("unsupported typed current-view schema")
    source_hashes = verify_sources(root)
    raw = [row for table in inventory.TABLES for row in inventory.read_table(root, table)]
    by_key = {row["report_key"]: row for row in raw}
    current_policy, policy_hash = inventory.policy_guards(root)
    strict = {}
    for entry in view["strict"]:
        key = inventory.canonical(normal_signature(entry["signature"]))
        if key in strict or not entry["model_id"] or not entry["certificate_declaration"]:
            raise ValueError("duplicate or incomplete typed strict entry")
        entry["instance_id"] = instance_id(entry["signature"])
        strict[key] = entry
    seen, strict_used, empirical, all_instances = set(), set(), set(), set()
    kinds = Counter()
    for exported in view["historical_rows"]:
        key = exported["report_key"]
        if key in seen or key not in by_key:
            raise ValueError("unknown/duplicate historical source in current view")
        seen.add(key)
        row = by_key[key]
        if any(exported[k] != row[k] for k in ("table", "artifact", "format")):
            raise ValueError("current view altered historical row provenance")
        expected = {"rule_id": row["rule_id"], "precision": row["precision"],
                    "guards": (current_policy[row["rule_id"]] if row["table"] == "ReportedAdmission"
                               else row["guards"]), "syntax_domain": None}
        if row["rule_id"] == "COUNT-ZERO":
            expected["syntax_domain"] = "constant int32 zero"
        elif row["rule_id"] == "COUNT-SUCCESSOR":
            expected["syntax_domain"] = (f"0 <= i < {row['syntax_domain']['upper_exclusive']}; "
                                         "int32 i+1 cannot overflow")
        if normal_signature(exported["signature"]) != expected:
            raise ValueError(f"current view precision/domain mismatch: {key}")
        signature = inventory.canonical(expected)
        all_instances.add(signature)
        expected_kind = "strict-model-proof" if signature in strict else "empirical"
        if exported["evidence_kind"] != expected_kind:
            raise ValueError("row exclusion does not match a typed strict source")
        kinds[expected_kind] += 1
        if signature in strict:
            strict_used.add(signature)
        else:
            empirical.add(signature)
        # Reference existing evidence; do not copy the report payload.
        exported["declaration"] = row["declaration"]
        exported["evidence_sha256"] = row["report_sha256"]
        exported["instance_id"] = instance_id(expected)
    if seen != set(by_key) or strict_used != set(strict):
        raise ValueError("incomplete historical coverage or unmatched strict source")
    expected_counts = {
        "historical_rows": len(raw), "historical_instances": len(all_instances),
        "strict_instances": len(strict), "strict_historical_rows": kinds["strict-model-proof"],
        "empirical_historical_rows": kinds["empirical"], "empirical_instances": len(empirical),
    }
    if view["counts"] != expected_counts:
        raise ValueError("Lean-exported inventory counts disagree with exact source join")
    view["empirical_instance_ids"] = sorted(instance_id(json.loads(key)) for key in empirical)
    view["verification"] = {
        "historical_evidence_unchanged": True,
        "source_domain_policy_sha256": policy_hash,
        "source_files_sha256": source_hashes,
        "empirical_rule_ids": dict(sorted(Counter(json.loads(k)["rule_id"] for k in empirical).items())),
        "guard_provenance": "Legacy raw rows use current source/policy domains; this is not historical sampler evidence.",
        "model_scope": "Strict certificates hold in their named software models; hardware bindings and actual kernel use are separate.",
        "new_experiments": 0,
    }
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--lean", type=Path)
    parser.add_argument("--output", type=Path, help="write only this new current-view file; default stdout")
    parser.add_argument("--summary", action="store_true", help="print counts and validation summary only")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        view, build = lean_view(root, args.receipt, args.lean)
        checked = checked_view(root, view)
        checked["build_validation"] = build
        result = ({"counts": checked["counts"], "verification": checked["verification"],
                   "checked_source_receipts": len(build.get("checked_source_receipts", {}))}
                  if args.summary else checked)
        output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            destination = args.output.resolve()
            protected = {root / "VeriTile/Triton/Float" / (table + ".lean")
                         for table in inventory.TABLES}
            report_directories = {(root / row["artifact"].split("#")[0]).parent
                                  for row in checked["historical_rows"]}
            if destination in protected or any(destination.is_relative_to(p) for p in report_directories):
                raise ValueError("current-view output must not overwrite historical tables or report directories")
            args.output.write_text(output)
        else:
            print(output, end="")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
