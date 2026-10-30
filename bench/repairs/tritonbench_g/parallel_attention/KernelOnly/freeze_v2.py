"""Additive v2 freeze: validate immutable v1 and all new scoped build receipts."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import freeze

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
V1_SHA = "cd81f2de8b72819da8dfa93ff6d8140fa45fc73ec154dc0dd84dcc887cd4f2cb"


def main():
    target = HERE / sys.argv[1]
    assert not target.exists(), f"Refusing to replace a frozen packet: {target}"
    v1_path = HERE / "packet-v1.json"
    assert freeze.sha(v1_path) == V1_SHA, "v1 manifest changed"
    v1 = json.loads(v1_path.read_text())
    for name, expected in v1["file_sha256"].items():
        assert freeze.sha(ROOT / name) == expected, f"v1 file changed: {name}"
    object_count = 0
    for module in v1["modules"].values():
        for name, expected in module["objects"].items():
            assert freeze.sha(Path(name)) == expected, f"v1 object changed: {name}"
            object_count += 1
    argv = sys.argv
    with tempfile.TemporaryDirectory(prefix="parallel-attention-v2-freeze-") as directory:
        temporary = Path(directory) / "checked.json"
        try:
            sys.argv = [str(HERE / "freeze.py"), str(temporary), *argv[2:]]
            with contextlib.redirect_stdout(io.StringIO()):
                freeze.main()
        finally:
            sys.argv = argv
        packet = json.loads(temporary.read_text())
    packet["schema"] = 2
    packet["entry"] = str((HERE / "PublicV2.lean").relative_to(ROOT))
    packet["parent_packet"] = {"path": str(v1_path.relative_to(ROOT)), "sha256": V1_SHA,
                               "unchanged_files": len(v1["file_sha256"]), "unchanged_objects": object_count,
                               "v1_common_backing_gap": "corrected by additive joint endpoints"}
    packet["file_sha256"][str(v1_path.relative_to(ROOT))] = V1_SHA
    packet["new_modules"] = sorted(set(packet["modules"]) - set(v1["modules"]))
    packet["reused_module_count"] = len(v1["modules"])
    packet["scope"]["joint_physical_post"] = True
    packet["scope"]["common_initial_final_input_output_backing"] = True
    packet["scope"]["physical_writer_uniqueness"] = True
    packet["scope"]["input_input_aliases_permitted"] = True
    with target.open("x") as stream:
        stream.write(json.dumps(packet, indent=2) + "\n")
    print(json.dumps({"packet": str(target), "sha256": freeze.sha(target),
                      "module_count": packet["module_count"], "new_module_count": len(packet["new_modules"]),
                      "source_dependency_count": packet["repository_source_dependency_count"],
                      "v1_unchanged_files": len(v1["file_sha256"]), "v1_unchanged_objects": object_count}, indent=2))


if __name__ == "__main__":
    main()
