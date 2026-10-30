"""Freeze this complete selected-source task, preserving previous task records."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / "bench/repairs/tritonbench_g/matmul_triton1"
OUT = ROOT / ".veritile-cache/task-deliveries/matmul_triton1/authorized-resumption-v1"
BUILD = ROOT / ".veritile-cache/kernel-build/lib/lean"
SOURCE_SHA = "705bfb2cecafd2644cfe60922a0eebce2e110d0221343c7e87b273c14bbfe62a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    body = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        assert path.read_text() == body, ("Refusing to replace changed evidence", path)
        return
    with path.open("x") as stream:
        stream.write(body)


def main():
    frozen = OUT / "frozen"
    assert not frozen.exists(), "Refusing to replace an immutable packet"
    selected = TASK / "revision001/matmul_triton1_repaired.py"
    assert sha(selected) == SOURCE_SHA
    before = json.loads((OUT / "preservation-before.json").read_text())
    allowed = {str((TASK / name).relative_to(ROOT)) for name in
               ["Kernels.lean", "Correct.lean", "FPEquiv.lean", "README.md"]}
    changed, preserved = [], []
    for name, digest in before.items():
        current = sha(ROOT / name)
        if current != digest:
            assert name in allowed, ("preservation drift", name)
            changed.append({"path": name, "before": digest, "after": current})
        else:
            preserved.append({"path": name, "sha256": digest})
    assert not (TASK / "matmul_triton1_candidate.py").exists()
    save(OUT / "preservation.json", {"status": "passed", "preserved": preserved,
         "changed": changed, "original_and_selected_python_unchanged": True,
         "revision001_unchanged": True, "previous_rejection_record_preserved": True})

    tree = ast.parse(selected.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "matmul_kernel")
    dots = [n for n in ast.walk(kernel) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "dot"]
    assert len(dots) == 1 and any(k.arg == "allow_tf32" and isinstance(k.value, ast.Constant)
                                 and k.value.value is False for k in dots[0].keywords)
    loads = [n for n in ast.walk(kernel) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "load"]
    stores = [n for n in ast.walk(kernel) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "store"]
    assert len(loads) == 2 and len(stores) == 1
    assert all(any(k.arg == "mask" for k in n.keywords) for n in loads + stores)
    save(OUT / "source-binding.json", {"status": "passed", "source": str(selected),
         "source_sha256": SOURCE_SHA, "python_candidate_created": False,
         "float_transformation_added": False, "masked_loads": 2, "masked_stores": 1,
         "allow_tf32": False, "quotation_identity": "Kernel.source_eq",
         "finite64_source_correspondence": "Revision001.FiniteSource.program_source",
         "whole_grid_memory_identity": "Kernel.execute",
         "physical_store_correspondence": "Kernel.program_physical",
         "scope": "Existing selected live carry graph; unchanged revision001 proof definitions; new actual Tensor/common backing endpoints."})

    pins = set()
    owned = sorted([*TASK.glob("*.lean"), *(TASK / "Proofs").glob("*.lean")])
    assert len(owned) == 11
    builds = []
    for source in owned:
        code = re.sub(r"/-.*?-/", "", source.read_text(), flags=re.S)
        code = re.sub(r"--[^\n]*", "", code)
        assert not re.search(r"\b(sorry|admit)\b|^\s*axiom\s", code, re.M), source
        receipt_path = BUILD / source.relative_to(ROOT).with_suffix(".workspace-build.json")
        receipt = json.loads(receipt_path.read_text())
        assert receipt["source_sha256"] == sha(source) and receipt["exit_code"] == 0, source
        for name, digest in receipt["objects"].items():
            path = Path(name)
            assert sha(path) == digest, path
            pins.add(path)
        pins.update([source, receipt_path])
        builds.append({"source": str(source), "source_sha256": sha(source),
                       "receipt": str(receipt_path), "receipt_sha256": sha(receipt_path),
                       "objects": receipt["objects"]})

    imports = json.loads((OUT / "import-input.json").read_text())
    headers = {r["path"]: r["imports"] for r in map(json.loads, (OUT / "binary-imports.jsonl").read_text().splitlines())}
    no_receipts, inherited_receipts = [], []
    for row in imports["repository"]:
        source = Path(row["source"])
        assert headers[row["resolved_olean"]] == ["Init", *row["source_imports"]], row["module"]
        pins.add(source)
        receipt_path = Path(row["effective_olean"]).with_suffix(".workspace-build.json")
        if source not in owned:
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["source_sha256"] == sha(source) and receipt["exit_code"] == 0, row
                for name, digest in receipt["objects"].items():
                    assert sha(Path(name)) == digest, name
                    pins.add(Path(name))
                pins.add(receipt_path)
                inherited_receipts.append(row["module"])
            else:
                no_receipts.append(row["module"])
    for row in imports["repository"] + imports["external_frontier"]:
        path = Path(row["resolved_olean"])
        assert Path(row["effective_olean"]).resolve() == path
        for extension in [".olean", ".olean.private", ".olean.server", ".ilean", ".trace"]:
            file = path.with_suffix(extension)
            if file.is_file():
                pins.add(file)
    save(OUT / "import-audit.json", {"status": "passed", "repository_source_count": len(imports["repository"]),
         "binary_headers_checked": len(headers), "source_binary_import_differences": [],
         "external_frontier_count": len(imports["external_frontier"]),
         "fresh_task_receipts": len(builds), "inherited_current_receipts": inherited_receipts,
         "inherited_without_workspace_receipts": no_receipts,
         "boundary": "Inherited cached objects are pinned at their resolved paths and reused; import-header agreement is checked, not a fresh rebuild of their source."})

    profile = json.loads((OUT / "default-profile/receipt.json").read_text())
    assert profile["status"] == "pass" and profile["source_sha256"] == SOURCE_SHA
    assert [r["abi"] for r in profile["rows"]] == ["i32", "i64"]
    for row in profile["rows"]:
        assert row["num_stages"] == 3 and row["num_warps"] == 4
        assert len(row["fma_chains"]) == 8
        assert all(c["terms"] == 32 and c["k_order"] == list(range(32)) and c["live_carry"] for c in row["fma_chains"])
        for name, digest in row["artifact_sha256"].items():
            assert sha(Path(name)) == digest
    assert profile["gpu_execution"] is False and profile["numerical_experiment"] is False
    pins.add(ROOT / "bench/tritonbench_g/matmul_triton1/matmul_triton1.py")
    pins.add(OUT.parent / "permission-blocker.json")
    pins.update(p for p in TASK.rglob("*") if p.is_file() and p.suffix in [".lean", ".py", ".md"])
    pins.update(p for p in OUT.rglob("*") if p.is_file())
    pins.update([ROOT / "lean-toolchain", ROOT / "lake-manifest.json",
                 ROOT / ".veritile-cache/build-tools/workspace_lean_local_first_v2.py",
                 ROOT / ".veritile-cache/build-tools/v2-root-review.json",
                 ROOT / ".veritile-cache/build-tools/v2-adoption.json"])
    frozen.mkdir()
    records = []
    for original in sorted(pins):
        destination = frozen / "files" / str(original).lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        digest = sha(original)
        assert sha(destination) == digest
        records.append({"path": str(original), "sha256": digest,
                        "snapshot": str(destination.relative_to(frozen))})
    packet = {"schema": 1, "task": "matmul_triton1", "status": "ready-for-independent-whole-task-review",
              "entry": str(TASK / "Public.lean"), "selected_source": str(selected),
              "selected_source_sha256": SOURCE_SHA, "python_candidate_created": False,
              "fresh_module_count": 11, "repository_source_dependency_count": len(imports["repository"]),
              "external_frontier_count": len(imports["external_frontier"]), "builds": builds,
              "preserved_file_count": len(preserved), "changed_preexisting_files": changed,
              "scope": {"kernel_complete_grid": True, "finite_typed_same_memory": True,
                        "real_independent_full_k_sum": True, "native_ascending_padded_scalar_fma": True,
                        "common_tensor_word_and_byte_post": True, "full_input_storage_and_exterior_frames": True,
                        "physical_writer_uniqueness": True, "wrapper": "existing assertions and metadata only",
                        "default_stages3_offline_abi_compiles": 2, "gpu_execution": False,
                        "new_numerical_experiments": False, "core_lake_ledger_edits": False,
                        "independent_review_completed": False},
              "boundaries": ["Ordinary Tensor/storage and successful representable launch contracts",
                             "Selected SM90 compiler lane/address/synchronization mapping",
                             "Native zero and single-FMA IEEE correspondence is explicit and conditional",
                             "Real arithmetic interpretation does not claim FP-to-Real identity",
                             "Inherited cached dependency objects are pinned; no fresh shared-core rebuild"],
              "historical_evidence": {"stages2": "Retained separately; does not certify default stages3",
                                      "cpu_cases": "Saved 27 rejection and 16 CPU-interpreted matmul1 cases; not rerun"},
              "files": records}
    save(frozen / "packet.json", packet)
    archive = OUT / "packet.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(frozen, arcname="matmul_triton1")
    save(OUT / "delivery.json", {"status": packet["status"], "packet": str(frozen / "packet.json"),
         "packet_sha256": sha(frozen / "packet.json"), "archive": str(archive), "archive_sha256": sha(archive),
         "pin_count": len(records), "fresh_module_count": 11, "repository_sources": len(imports["repository"]),
         "preserved_files": len(preserved), "independent_review_completed": False})
    print((OUT / "delivery.json").read_text())


if __name__ == "__main__":
    main()
