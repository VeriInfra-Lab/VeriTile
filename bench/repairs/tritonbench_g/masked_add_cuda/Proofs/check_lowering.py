"""Inspect both actual addition orders offline; no GPU or numerical experiment."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


def instructions(ptx):
    """Keep executable PTX lines, excluding source locations and declarations."""
    output = []
    for line in ptx.splitlines():
        line = line.strip()
        if not line or line.startswith(("//", ".", "{", "}")) or line.endswith(":"):
            continue
        if re.match(r"(?:@!?%\w+\s+)?[a-z][\w.]*(?:\s|;)", line) and ";" in line:
            line = line.replace("masked_add_commuted_kernel_param_", "kernel_param_")
            line = line.replace("masked_add_kernel_param_", "kernel_param_")
            output.append(line)
    return output


def main():
    here = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get("MASKED_ADD_LOWERING_DIR", "/tmp/veritile-masked-add-candidate-lowering"))
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "cache")
    source = here / "masked_add_cuda_repaired.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "masked_add_kernel")
    candidate = copy.deepcopy(node)
    candidate.name = "masked_add_commuted_kernel"
    update = candidate.body[-2]
    assert isinstance(update, ast.AugAssign) and isinstance(update.op, ast.Add)
    assert isinstance(update.target, ast.Name) and update.target.id == "grad"
    candidate.body[-2] = ast.Assign(targets=[ast.Name(id="grad", ctx=ast.Store())],
                                   value=ast.BinOp(left=update.value, op=ast.Add(),
                                                  right=ast.Name(id="grad", ctx=ast.Load())))
    module = ast.fix_missing_locations(ast.Module(body=[node, candidate], type_ignores=[]))
    module_path = output / "sources.py"
    module_path.write_text("import triton\nimport triton.language as tl\n" + ast.unparse(module) + "\n")
    spec = importlib.util.spec_from_file_location("masked_add_lowering_sources", module_path)
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    report = {"triton": triton.__version__, "gpu_execution": False,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "profiles": []}
    for arch in (80, 90):
        for mask_type in ("i32", "i64"):
            signature = {"grad_ptr": "*fp32", "p_ptr": "*fp32", "p_mask_ptr": "*" + mask_type,
                         "n_elements": "i32", "alpha": "fp32", "BLOCK_SIZE": "constexpr"}
            streams = []
            for variant, fn in (("original", imported.masked_add_kernel),
                                ("commuted", imported.masked_add_commuted_kernel)):
                result = triton.compile(ASTSource(fn, signature, constexprs={"BLOCK_SIZE": 1024}),
                                        target=GPUTarget("cuda", arch, 32),
                                        options={"enable_fp_fusion": True, "num_warps": 4})
                ptx = result.asm["ptx"]
                stem = f"sm{arch}_{mask_type}_{variant}"
                (output / f"{stem}.ptx").write_text(ptx)
                (output / f"{stem}.ttir").write_text(result.asm["ttir"])
                stream = instructions(ptx)
                streams.append(stream)
                assert sum("fma.rn.f32" in line for line in stream) == 8
                assert not any(re.search(r"(?:mul|add)(?:\.rn)?\.f32", line) for line in stream)
            assert streams[0] == streams[1], f"different emitted instruction streams: sm{arch} {mask_type}"
            report["profiles"].append({"arch": arch, "mask": mask_type, "fp_fusion": True,
                "same_instructions": True, "fma_count": 8,
                "instruction_sha256": hashlib.sha256("\n".join(streams[0]).encode()).hexdigest()})
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
