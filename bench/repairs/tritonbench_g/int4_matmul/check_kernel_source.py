"""Exact canonical wrapper arguments and native JIT specializations, offline.

Real CPU storage backs input metadata; CUDA allocation/context/launch are mocked.
No floating approximation experiment or GPU execution is claimed.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "int4_matmul_repaired.py"
SOURCE_SHA = "00132e33a59c145f856047e40fcce4a096105815c080ea0d324553ac6cf1a8ae"


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(output, packed_only=False):
    assert pin(SOURCE)["sha256"] == SOURCE_SHA
    os.environ.pop("TRITON_INTERPRET", None)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "source-check.json").exists(), "preserve completed evidence; use a new output directory"
    os.environ["TRITON_CACHE_DIR"] = str(output / "compiler-cache")
    import torch
    import triton
    from triton.runtime.jit import create_function_from_signature
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource
    checked = load(HERE / "check_contract.py", "int4_source_guard_helpers")
    module = load(SOURCE, "int4_source_canonical")
    kernel = module.matmul_kernel.fn
    configs = module.matmul_kernel.configs
    Tensor = checked.Tensor
    assert len(configs) == 36
    calls = {}

    class Launch:
        def __getitem__(self, grid):
            calls["grid"] = grid

            def invoke(*args):
                calls["args"] = args

            return invoke

    def zeros(*shape, device=None, dtype=None):
        dims = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return Tensor(torch.zeros(dims, dtype=dtype), device or "cuda:0")

    def where(condition, left, right):
        return Tensor(torch.where(checked.unwrap(condition), checked.unwrap(left), checked.unwrap(right)), condition.device)

    fake = SimpleNamespace(
        Tensor=Tensor, FloatTensor=Tensor, IntTensor=Tensor, strided=torch.strided,
        float32=torch.float32, int32=torch.int32,
        zeros=zeros, where=where, isfinite=lambda t: Tensor(torch.isfinite(t.data), t.device),
        cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()),
    )
    tree = ast.parse(SOURCE.read_text())
    names = {"_ordinary", "_independent", "_validate_packed", "_validate_matmul", "matmul_dequantize_int4_s2",
             "_validate_quantize", "quantize_int4"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": fake, "triton": triton, "matmul_kernel": Launch()}
    exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), namespace)
    target = GPUTarget("cuda", 90, 32)
    backend = CUDABackend(target)
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    profiles = []
    cases = [("canonical_singleton", 1, 8, 8, 8, False, (0, 9)),
             ("packed_singleton", 1, 8, 8, 8, True, (5, 14)),
             ("unit_group_tail", 3, 40, 16, 1, False, (0,))]
    for label, m, k, n, group, packed, indices in cases:
        if packed_only and not packed:
            continue
        a = torch.ones((m, k), dtype=torch.float32)
        q = torch.full((k // 8, n), 0x11111111, dtype=torch.int32)
        scales = torch.ones((k // group, n), dtype=torch.float32)
        z = torch.zeros((k // group, n // 8), dtype=torch.int32)
        out = torch.full((m, n), 7, dtype=torch.float32)
        if packed:
            # Obtain these tensors from the actual quantizer AST. On singleton
            # dimensions .T.contiguous().T can silently retain canonical strides.
            weight = Tensor(torch.linspace(-3, 5, k * n, dtype=torch.float32).reshape(k, n))
            qw, scale, qz, returned_group = namespace["quantize_int4"](weight, group, 0)
            assert returned_group == group
            q, scales, z = (value.data for value in (qw, scale, qz))
            assert q.shape == scales.shape == (1, 8) and z.shape == (1, 1)
            assert q.stride() == scales.stride() == z.stride() == (1, 1)
        wrapped = list(map(Tensor, (a, q, scales, z, out)))
        result = namespace["matmul_dequantize_int4_s2"](*wrapped[:4], group, wrapped[4])
        assert result is wrapped[4] and out.eq(0).all().item()
        args = [x.data if isinstance(x, Tensor) else x for x in calls["args"]]
        assert len(args) == 19 and args[5:8] == [m, n, k]
        if packed:
            assert args[8:18] == [8, 1, 1, 1, 8, 1, 1, 1, 1, 1]
        print(f"canonical assertions admit {label}; exact actual strides captured", flush=True)
        for index in indices:
            cfg = configs[index]
            options = dict(num_warps=cfg.num_warps, num_stages=cfg.num_stages, num_ctas=cfg.num_ctas)
            # Use the same generated binder as JITFunction.run. _pack_args uses
            # bound.values() positionally; config dictionary insertion order is
            # not kernel parameter order and must never construct this mapping.
            bound, special, bound_options = binder(*args, **cfg.kwargs, **options)
            assert list(bound) == kernel.arg_names
            assert bound_options == options
            for name, value in zip(kernel.arg_names[:5], args[:5]):
                assert bound[name] is value, name
            scalar_values = dict(zip(kernel.arg_names[5:19], args[5:19]))
            scalar_values.update(cfg.kwargs)
            for name, value in scalar_values.items():
                assert bound[name] == value, (name, bound[name], value)
            packed_options, signature, constants, attrs = kernel._pack_args(
                backend, bound_options, bound, special, None)
            assert list(signature) == kernel.arg_names
            named_constants = {kernel.arg_names[key[0]]: value for key, value in constants.items()}
            assert len(named_constants) == len(constants)
            for name, value in scalar_values.items():
                ordinal = kernel.arg_names.index(name)
                if signature[name] == "constexpr":
                    assert named_constants[name] == value, (name, named_constants[name], value)
                    assert special[ordinal] == ("constexpr", value)
                else:
                    assert name not in named_constants and special[ordinal][0] == signature[name]
            for name, value in cfg.kwargs.items():
                assert signature[name] == "constexpr" and named_constants[name] == value
            for name, value in options.items():
                assert getattr(packed_options, name) == value
            assert signature["stride_ak"] == "constexpr"
            if m == 1:
                assert signature["M"] == "constexpr"
            if group == 1:
                assert signature["group_size"] == "constexpr"
            if packed:
                for name in ("stride_bk", "stride_bsk"):
                    assert signature[name] == "constexpr" and named_constants[name] == 1
            compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                                      target=target, options=options)
            artifacts = []
            for ext in ("ttir", "ttgir", "llir", "ptx"):
                path = output / f"{label}_config{index}.{ext}"
                path.write_text(compiled.asm[ext])
                artifacts.append(pin(path))
            ptx = compiled.asm["ptx"]
            instructions = re.findall(r"(?:wgmma\.mma_async|mma\.sync)[^;]+;", ptx)
            assert instructions
            assert all("tf32" in instruction and "f32" in instruction and
                       re.search(r"m\d+n\d+k8\.", instruction) for instruction in instructions)
            ttir = compiled.asm["ttir"]
            bm, bn, bk = (cfg.kwargs[f"BLOCK_SIZE_{axis}"] for axis in "MNK")
            dots = [line.strip() for line in ttir.splitlines() if " = tt.dot " in line]
            assert len(dots) == 1
            dot = re.search(r"(%[\w]+) = tt.dot [^,]+, [^,]+, (%[\w]+), inputPrecision = tf32", dots[0])
            assert dot and f"tensor<{bm}x{bk}xf32> * tensor<{bk}x{bn}xf32> -> tensor<{bm}x{bn}xf32>" in dots[0]
            loop = next(line.strip() for line in ttir.splitlines() if " = scf.for " in line)
            assert dot[2] + " = " in loop
            assert any(dot[1] in line for line in ttir.splitlines() if "scf.yield" in line)
            assert "arith.addf" not in ttir
            split = cfg.kwargs["SPLIT_K"]
            assert ("tt.store " in ttir) == (split == 1)
            assert ("tt.atomic_rmw fadd" in ttir) == (split != 1)
            assert ("st.global" in ptx) == (split == 1)
            assert bool(re.search(r"(?:atom|red)\.global[^\s;]*\.add\.f32", ptx)) == (split != 1)
            profiles.append({"case": label, "shape": {"M": m, "N": n, "K": k, "group_size": group},
                             "config_index": index, "config": cfg.kwargs, "options": options,
                             "actual_strides": args[8:18], "grid": calls["grid"](cfg.kwargs),
                             "packed_input_origin": "actual canonical quantize_int4 AST outputs" if packed else "canonical CPU allocation",
                             "signature": signature,
                             "binding": {"implementation": "create_function_from_signature + JITFunction._pack_args",
                                         "parameter_order": list(bound), "all_scalar_values": scalar_values,
                                         "all_scalar_and_config_values_asserted": True,
                                         "all_packed_constexpr_values_asserted": True},
                             "constants": named_constants,
                             "attrs": {kernel.arg_names[key[0]]: value for key, value in attrs.items()},
                             "instruction_sites": len(instructions),
                             "instruction_families": sorted(set(i.split(" ", 1)[0] for i in instructions)),
                             "native_structure": {"ttir_dot": dots[0], "ttir_loop": loop,
                                                  "live_loop_carry_used_and_yielded": True,
                                                  "standalone_accumulator_add": False,
                                                  "fixed_instruction_k": 8,
                                                  "epilogue": "store" if split == 1 else "atomic_add"},
                             "artifacts": artifacts})
            print(f"{label} config {index}: native specialization compiles; {len(instructions)} TF32 sites", flush=True)
    receipt = {"task": "int4_matmul", "evidence_round": 3,
               "source": pin(SOURCE), "checker": pin(Path(__file__)),
               "guard_helper": pin(HERE / "check_contract.py"), "diagnostic_exit_code": 0,
               "triton_version": triton.__version__, "compiler_target": "cuda sm90 warp32",
               "profiles": profiles, "canonical_source_unchanged": True,
               "GPU_execution": False, "independent_numerical_replay": False,
               "boundary": "Canonical wrapper assertions/argument capture using full CPU backing with mocked CUDA. Actual decorated kernel and native JIT specialization compiled offline. Saved instructions are compiler evidence, not hardware numerical validation."}
    path = output / "source-check.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"profiles": len(profiles), "receipt": pin(path)}), flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".veritile-cache/kernel-evidence/int4_matmul/exact-specialization-round3")
    parser.add_argument("--packed-only", action="store_true", help="replace only the two affected singleton profiles")
    args = parser.parse_args()
    run(args.output, args.packed_only)
