"""Validate current scoped Lean receipts and create an immutable review manifest."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CACHE = ROOT / ".veritile-cache/kernel-build/lib/lean"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = HERE / sys.argv[1]
    assert not target.exists(), f"Refusing to replace a frozen packet: {target}"
    from source_guard import check
    guard = check()
    files = {}
    modules = {}
    queue = list(HERE.glob("*.lean"))
    dependencies = set()
    while queue:
        path = queue.pop()
        if path in dependencies:
            continue
        dependencies.add(path)
        files[str(path.relative_to(ROOT))] = sha(path)
        for line in path.read_text().splitlines():
            if line.startswith("import "):
                for name in line.removeprefix("import ").split():
                    candidate = ROOT / (name.replace(".", "/") + ".lean")
                    if candidate.is_file():
                        queue.append(candidate)
    for source in sorted(HERE.glob("*.lean")):
        relative = source.relative_to(ROOT)
        receipt_path = CACHE / relative.with_suffix(".workspace-build.json")
        assert receipt_path.is_file(), f"Missing build receipt: {relative}"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["exit_code"] == 0 and receipt["source_sha256"] == sha(source), f"Stale build: {relative}"
        for name, expected in receipt["objects"].items():
            assert sha(Path(name)) == expected, f"Object drift: {name}"
        files[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
        modules[str(relative)] = {"source_sha256": sha(source), "receipt": str(receipt_path.relative_to(ROOT)),
                                  "receipt_sha256": sha(receipt_path), "objects": receipt["objects"]}
    saved = ["parallel_attention_repaired.py", "validation.json", "cuda-lowering.json",
             "lowering-binding.json", "checks/kernel-ast.json", "checks/COMPILER-CONTRACT.md",
             "checks/mutation-guards.json", "checks/resolve.py", "checks/check_dot.py",
             "checks/check_scalar.py", "checks/check_norm.py"]
    for name in saved:
        path = HERE.parent / name
        files[str(path.relative_to(ROOT))] = sha(path)
    for path in HERE.iterdir():
        if path.is_file() and (path.suffix in (".md", ".py") or path.name in ("source-pins.json", "source-guard-result.json")):
            files[str(path.relative_to(ROOT))] = sha(path)
    logs = {}
    for name in sys.argv[2:]:
        path = Path(name).resolve()
        body = path.read_text()
        assert "error:" not in body and "DISALLOWED" not in body, f"Failed build log supplied: {path}"
        logs[str(path)] = sha(path)
    packet = {"schema": 1, "task": "parallel_attention kernel-only",
              "status": "ready-for-independent-review", "entry": str((HERE / "Public.lean").relative_to(ROOT)),
              "module_count": len(modules), "repository_source_dependency_count": len(dependencies),
              "source_guard": guard, "modules": modules, "file_sha256": files, "build_log_sha256": logs,
              "scope": {"forward": "O and Z", "backward": "partial value-shard DQ/DK scratch and complete DV",
                        "real_scale": "actual ABI sigma, arbitrary Real",
                        "native": "fixed staged interpretation plus separately admitted local compiler/ABI mapping",
                        "host_sum_normalization_autograd_allocation_byteheap": False,
                        "numerical_or_compiler_experiments_rerun": False,
                        "independent_review_completed": False}}
    target.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps({"packet": str(target), "sha256": sha(target),
                      "module_count": len(modules), "source_dependency_count": len(dependencies)}, indent=2))


if __name__ == "__main__":
    main()
