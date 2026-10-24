"""Offline controls for the unchanged source's ordinary specialized SM90 ABI.

This records compiler structure; it does not certify the whole lowering plan
or measure numerical agreement of native instructions.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = HERE.parent / "context_attn_llama_repaired.py"
OUT = ROOT / ".veritile-cache/kernel-evidence/context_attn_llama"


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run():
    assert pin(SOURCE)["sha256"] == "8317f23a4c8c04305b230369c2f59f6c4aaf41a5e80dca5336e88a1e13617e0b"
    os.environ.pop("TRITON_INTERPRET", None)
    os.environ["TRITON_CACHE_DIR"] = str(OUT / "compiler-cache")
    temporary = OUT / "compiler-tmp"
    temporary.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    import torch
    import triton
    from triton._C.libtriton import native_specialize_impl
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource

    spec = importlib.util.spec_from_file_location("nopad_kernel_compile_control", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module._fwd_kernel
    node = next(x for x in ast.parse(SOURCE.read_text()).body
                if isinstance(x, ast.FunctionDef) and x.name == "_fwd_kernel")
    actual = ast.parse(kernel.src).body[0]
    assert ast.dump(actual.args) == ast.dump(node.args)
    assert [ast.dump(x) for x in actual.body] == [ast.dump(x) for x in node.body]
    target = GPUTarget("cuda", 90, 32)
    backend = CUDABackend(target)
    records = []
    for dimension in (16, 32, 64, 128, 256):
        for tile in (64, 128):
            offset=0 if tile==64 else 1
            heads,kvheads=4,2
            payloads = [torch.empty(130 * heads * dimension + 2, dtype=torch.float32)[offset:]
                        for _ in range(4)]
            metadata = [torch.empty(8,dtype=torch.int32)[offset:] for _ in range(4)]
            pages=torch.empty(4*256+2,dtype=torch.int32)[offset:]
            assert all(x.data_ptr()%16==4*offset for x in payloads+metadata+[pages])
            strides=[heads*dimension,dimension,1]
            kvstrides=[kvheads*dimension,dimension,1]
            arguments=[*payloads[:3],1.0/(dimension**.5)*1.4426950408889634,payloads[3],
                metadata[0],metadata[1],pages,metadata[2],*strides,*kvstrides,*kvstrides,*strides,
                256,1,heads//kvheads,metadata[3],heads,dimension,tile,tile]
            bound=dict(zip(kernel.arg_names,arguments,strict=True))
            specialization = [("constexpr", bound[p.name]) if p.is_constexpr else
                native_specialize_impl(backend, bound[p.name], p.is_const, not p.do_not_specialize,
                                       not p.do_not_specialize_on_alignment) for p in kernel.params]
            options = {"num_warps": 4 if dimension <= 64 else 8, "num_stages": 1, "num_ctas": 1}
            _, signature, constants, attrs = kernel._pack_args(backend, options, bound, specialization, None)
            compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                                      target=target, options=options)
            artifacts = []
            for ext in ("ttir", "ttgir", "llir", "ptx"):
                path = OUT / f"source_bm{tile}_d{dimension}_align{16 if offset == 0 else 4}.{ext}"
                path.write_text(compiled.asm[ext])
                artifacts.append(pin(path))
            ptx = compiled.asm["ptx"]
            assert "tf32" in compiled.asm["ttir"]
            assert "wgmma.mma_async.sync.aligned.m64n" in ptx
            assert "ex2.approx" in ptx and "fma.rn.f32" in ptx
            records.append({"dimension": dimension, "tile": tile, "heads": heads, "kv_heads": kvheads, "storage_offset": offset,
                "pointer_alignment_class": 16 if offset == 0 else 4,
                "signature": signature, "constants": {kernel.arg_names[k[0]]: v for k, v in constants.items()},
                "attrs": {kernel.arg_names[k[0]]: v for k, v in attrs.items()}, "options": options,
                "shared_memory_bytes": compiled.metadata.shared,
                "fixed_SM90_WGMMA_sites": ptx.count("wgmma.mma_async.sync.aligned.m64n"),
                "exp2_sites": ptx.count("ex2.approx"), "fma_sites": ptx.count("fma.rn.f32"),
                "actual_specialization_compile_success": True, "artifacts": artifacts})
            print(f"PASS SM90 BM={tile}, D={dimension}, heads={heads}, pointer offset={offset}", flush=True)
    report = {"source": pin(SOURCE), "checker": pin(Path(__file__)), "target": "CUDA SM90 warp32",
        "triton_version": triton.__version__, "torch_version": torch.__version__, "cases": records,
        "GPU_execution": False, "native_numerical_conformance": False,
        "whole_lowering_plan_certified": False,
        "scope": "Native specialization binder applied to the unchanged wrapper ABI, using CPU pointer handles for dtype/alignment only. All ten admitted tile/width combinations and aligned/four-byte-offset pointer classes compile with unit strides specialized normally. This is not a GPU launch or a proof of the complete physical reduction/operand-routing profile."}
    receipt = OUT / "compile-controls.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"offline_compiles": len(records), "receipt": pin(receipt)}), flush=True)


if __name__ == "__main__":
    run()
