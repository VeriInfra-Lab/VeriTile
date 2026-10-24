"""Freeze verified candidate-only evidence and its dependency closure; no builds."""
from pathlib import Path
import hashlib
import json
import re

TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[3]
TASK = TASK_DIR.relative_to(ROOT)
D = TASK_DIR / "kernel-evidence"
OUT = ROOT / ".veritile-cache/kernel-build/lib/lean"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    path = path if path.is_absolute() else ROOT / path
    assert path.is_file(), path
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha(path)}


predecessor = json.loads((D / "initial-pins.json").read_text())
for name, digest in predecessor.items():
    assert sha(ROOT / name) == digest, name

source = json.loads((D / "source-validation.json").read_text())
binder = json.loads((D / "binder-validation.json").read_text())
assert source["full_kernel_AST_equal"] and source["top_level_statements"] == 37
assert source["dsl_sha256"] == sha(TASK_DIR / "Kernels.lean")
assert source["source_sha256"] == sha(TASK_DIR / "int8_dequant_matmul_scalar_cast.py")
assert binder["source"]["sha256"] == source["source_sha256"]
assert binder["checker"]["sha256"] == sha(TASK_DIR / "check_binder.py")
assert binder["source_audit"]["guard_and_wrapper_AST_unchanged"]
assert binder["source_audit"]["configurations"] == 418
assert len(binder["source_audit"]["only_changes"]) == 9
assert binder["no_specialization_disabled"] and len(binder["cases"]) == 10
assert not binder["GPU_execution"] and not binder["actual_JIT_launch"]
for case in binder["cases"]:
    assert case["compilation"] == "passed", case["name"]
    for item in case["artifacts"]:
        assert sha(ROOT / item["path"]) == item["sha256"], item

modules = sorted(str(p.relative_to(ROOT)) for p in TASK_DIR.glob("*.lean"))
modules += sorted(str(p.relative_to(ROOT)) for p in (TASK_DIR / "Proofs").glob("*.lean"))
receipts = {}
for name in modules:
    path = Path(name)
    receipt_path = OUT / path.with_suffix(".workspace-build.json")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["source"] == name and receipt["source_sha256"] == sha(ROOT / path), name
    assert receipt["exit_code"] == 0 and receipt["objects"], name
    for name_obj, digest in receipt["objects"].items():
        obj = Path(name_obj)
        assert obj.is_relative_to(OUT) and not obj.is_symlink() and sha(obj) == digest, obj
    receipts[name] = {"receipt": record(receipt_path), "build": receipt}
    assert not re.search(r"\b(sorry|admit)\b", (ROOT / path).read_text()), path

audit = (D / "combined-audit.log").read_text()
assert ": error:" not in audit and "DISALLOWED axioms" not in audit
axioms = audit.count("axiom footprint ⊆ standard base ✓")
noncircular = audit.count("non-circular ✓")
assert axioms == 51 and noncircular == 9, (axioms, noncircular)
final_log = (D / "final-build.log").read_text()
assert ": error:" not in final_log and "DISALLOWED axioms" not in final_log
runs = json.loads((D / "validation-runs.json").read_text())
assert {item["session_id"] for item in runs["runs"]} == {80153, 66415, 28018}
assert all(item["exit_code"] == 0 and item["observed_by"] == "functions.write_stdin terminal result"
           for item in runs["runs"])

closure = {}
external = set()
pending = modules.copy()
while pending:
    name = pending.pop()
    if name in closure:
        continue
    path = ROOT / name
    obj = OUT / Path(name).with_suffix(".olean")
    assert path.is_file() and obj.is_file(), name
    family = sorted(p for p in obj.parent.glob(obj.stem + ".*") if p.is_file() and
                    (p.name.endswith(".ilean") or ".olean" in p.name))
    item = {"source_sha256": sha(path), "artifact": record(obj), "resolved_artifact": str(obj.resolve()),
            "artifact_family": [{**record(p), "resolved_path": str(p.resolve())} for p in family]}
    receipt_path = obj.with_suffix(".workspace-build.json")
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["source_sha256"] == sha(path) and receipt["exit_code"] == 0, receipt_path
        for name_obj, digest in receipt["objects"].items():
            assert sha(Path(name_obj)) == digest, name_obj
        item.update(receipt=record(receipt_path), status="current source/object workspace receipt verified")
    else:
        item["status"] = "existing dependency cache pinned; not rebuilt here"
    closure[name] = item
    for mod in re.findall(r"^import\s+([A-Za-z0-9_.]+)", path.read_text(), re.M):
        child = Path(mod.replace(".", "/") + ".lean")
        if (ROOT / child).is_file():
            pending.append(str(child))
        else:
            external.add(mod)
(D / "closure.json").write_text(json.dumps({"repository_imports": dict(sorted(closure.items())),
                                           "external_imports": sorted(external)}, indent=2) + "\n")

source_pins = {str(p.relative_to(ROOT)): sha(p) for p in sorted(TASK_DIR.glob("*"))
               if p.is_file() and p.suffix in {".lean", ".py", ".md"}}
source_pins.update({str(p.relative_to(ROOT)): sha(p) for p in sorted((TASK_DIR / "Proofs").glob("*.lean"))})
prefix = "VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast."
packet = {
    "task": "int8_dequant_matmul_scalar_cast",
    "status": "separate candidate complete; frozen for independent parent review",
    "blockers": [],
    "candidate_source": binder["source"],
    "predecessor_source": binder["baseline"],
    "predecessor_all_pins_preserved": predecessor,
    "python_delta": "Exactly nine scalar/stride .to(tl.int64) calls become tl.cast; all other bytes unchanged",
    "canonical_selected": False,
    "domain": "Original imported checked domain, all 418 configurations, unit/empty dimensions, K0, offsets and allowed aliases; no numeric restriction added",
    "prepared_state": "Fresh FP16 output; typed FP16 zeros for split-K>1, supplied by existing zero hook",
    "public_entries": {"Real": prefix + "correct", "native": prefix + "fp_equiv"},
    "scope": "Kernel-only readable-quotation execution, source store descriptor schedule, original-view result, output words/bytes and complete element/byte frame",
    "source_execution": {
        "full_quote_success": prefix + "Quote.fullResult",
        "actual_store_replay": prefix + "Quote.Result.replay_eq",
        "source_to_independent_reference": prefix + "Quote.sourceGrid_eq",
        "whole_quote_lowering_assumption": False,
        "numerical_result_assumptions": [],
        "concurrency": "Complete physical store lanes in arbitrary order; each descriptor is obtained by actual quotation execution; the full source store fold is proved equal to descriptor replay",
    },
    "model_boundaries": [
        "Fixed K32 MMA/WGMMA one-lane primitive realization, structural lane plan and fusion selection",
        "Ordinary local scalar/memory event ABI laws and original TensorView/storage laws",
        "Source-specific exact signed64 normalization uses the unchanged checked bounds",
        "Existing dependency objects pinned; no cold build or GPU execution claim",
    ],
    "native_real_equality": False,
    "deferred_numerical_experiments": [],
    "new_source_defects": [],
    "source_audit": {**source, "checker": record(TASK_DIR / "check_source.py"), "report": record(D / "source-validation.json")},
    "compiler_evidence": {"report": record(D / "binder-validation.json"), "binding": binder["binding"],
                          "cases": 10, "all_prior_failures_compile": True,
                          "native_specialization_preserved": True, "GPU_execution": False,
                          "artifacts": [a for c in binder["cases"] for a in c["artifacts"]]},
    "lean_checks": {"current_module_receipts": len(receipts), "axiom_checks": axioms,
                    "non_circular_checks": noncircular, "audit_source": record(TASK_DIR / "Audit.lean"),
                    "audit_log": record(D / "combined-audit.log"), "build_log": record(D / "final-build.log"),
                    "observed_session_exits": record(D / "validation-runs.json"),
                    "closure": record(D / "closure.json"), "repository_dependencies": len(closure),
                    "driver": record(ROOT / ".veritile-cache/build-tools/workspace_lean_local_first_v2.py"),
                    "driver_adoption": record(ROOT / ".veritile-cache/build-tools/v2-adoption.json")},
    "integration": {"parent_owns_ledger": True, "predecessor_core_and_lake_untouched": True},
    "source_pins": source_pins,
    "module_receipts": receipts,
    "freeze_script": record(Path(__file__).resolve()),
}
(D / "delivery.json").write_text(json.dumps(packet, indent=2) + "\n")
print(json.dumps({"delivery": str((D / "delivery.json").relative_to(ROOT)),
                  "sha256": sha(D / "delivery.json"), "module_receipts": len(receipts),
                  "repository_dependencies": len(closure), "predecessor_pins": len(predecessor)}, indent=2))
