"""Exact admitted-input counterexample for configuration-dependent INT32 carry.

Runs the canonical Python assertions with real CPU-backed content and mocked
CUDA allocation/launch, compiles only the two relevant canonical configurations
offline for SM90, and evaluates their documented K32 integer recurrences.
No CUDA kernel or floating numerical experiment is executed.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "int8_matmul_kernel_repaired.py"
ORIGINAL = ROOT / "bench/tritonbench_g/int8_matmul_kernel/int8_matmul_kernel.py"
SOURCE_SHA = "e95569108e080703b0bc8df8a61c862eba863cefe6fa4c6321c9ced9ca496cee"
K = 2**24 + 512
INT_MIN, INT_MAX = -(2**31), 2**31 - 1


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def signed32(value):
    value %= 2**32
    return value if value < 2**31 else value - 2**32


def recurrence(k, saturation):
    state = 0
    boundaries = []
    first_overflow = None
    updates = 0
    for field in range(4):
        weight = ((15 >> (2 * field)) & 3) - 1
        block_sum = 32 * (-128) * weight
        for _ in range(k // 128):
            exact = state + block_sum
            updates += 1
            if first_overflow is None and not INT_MIN <= exact <= INT_MAX:
                first_overflow = {"update": updates, "field": field, "seed": state, "exact": exact}
            state = max(INT_MIN, min(INT_MAX, exact)) if saturation else signed32(exact)
        boundaries.append(state)
    return {"output": state, "field_boundaries": boundaries,
            "instruction_updates": updates, "first_overflow": first_overflow}


def admission():
    import torch
    checked = load(HERE / "check_contract.py", "int8_counterexample_guard")
    a_data = torch.full((1, K), -128, dtype=torch.float32)
    b_data = torch.full((K // 4, 1), 15, dtype=torch.uint8)
    a = checked.Tensor(a_data.shape, dtype=a_data.dtype, strides=a_data.stride(), data=a_data)
    b = checked.Tensor(b_data.shape, dtype=b_data.dtype, strides=b_data.stride(), data=b_data)
    a.pointer, b.pointer = a_data.data_ptr(), b_data.data_ptr()
    call, runtime = checked.instantiate()
    original_empty = runtime.empty

    def empty(shape, device, dtype):
        result = original_empty(shape, device, dtype)
        result.data = torch.empty(shape, dtype=dtype)
        result.pointer = result.data.data_ptr()
        return result

    runtime.empty = empty
    result = call(a, b)
    assert result is runtime.output and runtime.events == ["allocation", "launch"]
    configurations = []
    for index, config in enumerate(runtime.configs):
        assert K % (4 * config.kwargs["BLOCK_SIZE_K"]) == 0
        assert runtime.grid(config.kwargs) == (1,)
        configurations.append({"index": index, "kwargs": config.kwargs,
                               "grid": [1], "static_assertion": True})
    assert len(configurations) == 15
    assert a_data.is_contiguous() and torch.isfinite(a_data).all().item()
    assert torch.trunc(a_data).eq(-128).all().item()
    return {
        "canonical_guard_accepted": True, "output_check_accepted": True,
        "real_CPU_backing_bytes": [a_data.nbytes, b_data.nbytes, result.data.nbytes],
        "actual_CPU_pointers": [a.pointer, b.pointer, result.pointer],
        "events": runtime.events, "all_configurations": configurations,
        "boundary": "Original assertion AST and complete CPU content; CUDA metadata/allocation/launch mocked. No GPU execution.",
    }


def compile_profiles(output):
    os.environ.pop("TRITON_INTERPRET", None)
    os.environ["TRITON_CACHE_DIR"] = str(output / "compiler-cache")
    import triton
    import torch
    from triton._C.libtriton import native_specialize_impl
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    module = load(SOURCE, "int8_counterexample_canonical")
    kernel = module.matmul_kernel.fn
    configs = module.get_autotune_config()
    target = GPUTarget("cuda", 90, 32)
    backend = CUDABackend(target)
    # These tiny CPU tensors are dtype/alignment tokens for the ordinary JIT
    # specializer, never kernel inputs. Admission above uses full witness data.
    # The actual CUDA witness can use ordinary aligned torch allocations.
    pointer_tokens = [torch.empty(1, dtype=dtype)
                      for dtype in (torch.float32, torch.uint8, torch.int32)]
    assert all(t.data_ptr() % 16 == 0 for t in pointer_tokens)
    profiles = []
    for index in (14, 4):
        config = configs[index]
        values = dict(zip(kernel.arg_names[:12],
                          [*pointer_tokens, 1, 1, K, K, 1, 1, 1, 1, 1]))
        values.update(config.kwargs)
        bound_args = {name: values[name] for name in kernel.arg_names}
        specialization = [
            ("constexpr", values[param.name]) if param.is_constexpr else
            native_specialize_impl(backend, values[param.name], param.is_const,
                                   not param.do_not_specialize,
                                   not param.do_not_specialize_on_alignment)
            for param in kernel.params
        ]
        options = {"num_warps": config.num_warps, "num_stages": config.num_stages,
                   "num_ctas": config.num_ctas}
        _, signature, constants, attrs = kernel._pack_args(
            backend, options, bound_args, specialization, None)
        assert signature["M"] == signature["N"] == "constexpr"
        compiled = triton.compile(ASTSource(kernel, signature=signature,
                                            constexprs=constants, attrs=attrs),
                                  target=target, options=options)
        paths = []
        for ext in ("ttir", "ttgir", "llir", "ptx"):
            path = output / f"config{index}.{ext}"
            path.write_text(compiled.asm[ext])
            paths.append(pin(path))
        ttir, llir, ptx = compiled.asm["ttir"], compiled.asm["llir"], compiled.asm["ptx"]
        dots = [line for line in ttir.splitlines() if "= tt.dot " in line]
        assert dots
        seeds = []
        for line in dots:
            seed = re.search(r"tt.dot %\w+, %\w+, (%\w+)", line).group(1)
            assert not seed.startswith("%cst"), line
            assert any(seed + " = " in header for header in ttir.splitlines() if "scf.for" in header), (seed, line)
            seeds.append(seed)
        assert "tt.elementwise_inline_asm \"add.u32" not in ttir
        # LLVM retains the source f32 -> signed i8 conversion. NVPTX lowers
        # that to rzi.s16.f32 followed by byte packing, not an s8 cvt opcode.
        # The witness -128 is exact in every intermediate representation.
        assert re.search(r"arith\.fptosi .*xf32> to tensor<.*xi8>", ttir)
        assert re.search(r"fptosi <\d+ x float> .* to <\d+ x i8>", llir)
        conversions = re.findall(r"cvt\.rzi\.s(?:16|32)\.f32[^;]+;", ptx)
        assert conversions
        if index == 14:
            instructions = re.findall(r"mma\.sync\.aligned\.m16n8k32\.row\.col\.satfinite\.s32\.s8\.s8\.s32[^;]+;", ptx)
            assert instructions and "wgmma.mma_async" not in ptx
            for instruction in instructions:
                registers = re.findall(r"\{([^}]+)\}", instruction)
                assert len(registers) == 4 and registers[0] == registers[3], instruction
            family = "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"
        else:
            instructions = re.findall(r"wgmma\.mma_async\.sync\.aligned\.m64n128k32\.s32\.s8\.s8[^;]+;", ptx)
            assert instructions and all("satfinite" not in line for line in instructions)
            ll_instructions = [line for line in llir.splitlines()
                               if 'asm sideeffect "wgmma.mma_async' in line]
            assert ll_instructions and all("i1 true)" in line for line in ll_instructions)
            for instruction in instructions:
                predicate = re.search(r",\s*(%p\d+);$", instruction).group(1)
                assignments = re.findall(r"^\s*\S+\s+" + re.escape(predicate)
                                         + r",\s*([^;]+);", ptx, re.MULTILINE)
                assert assignments == ["-1"], (predicate, assignments)
            family = "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8"
        profiles.append({"index": index,
                         "constants": {kernel.arg_names[key[0]]: value
                                       for key, value in constants.items()},
                         "signature": signature,
                         "attrs": {kernel.arg_names[key[0]]: value
                                   for key, value in attrs.items()},
                         "specialization": "Native Triton JIT specialization of the exact witness scalar arguments and aligned pointer dtypes; CPU pointer tokens used only to select dtype/alignment attributes.",
                         "options": options,
                         "target": "cuda sm90 warp32", "triton_version": triton.__version__,
                         "ttir_live_carry_seeds": seeds, "instruction_family": family,
                         "PTX_live_carry": "same destination/source registers" if index == 14
                                           else "scale-d predicate true in LLIR and PTX",
                         "FP32_to_signed8_lowering": {
                             "LLVM": "fptosi float to i8",
                             "PTX_conversion_sites": conversions,
                             "witness": "-128.0 converts exactly to signed16 -128; low byte 0x80 denotes signed8 -128",
                         },
                         "instruction_sites": len(instructions), "instructions": instructions,
                         "artifacts": paths})
        print(f"canonical config {index}: {family}, {len(instructions)} sites", flush=True)
    return profiles


def run(output):
    assert pin(SOURCE)["sha256"] == SOURCE_SHA
    original_ast, canonical_ast = [ast.parse(path.read_text()) for path in (ORIGINAL, SOURCE)]
    unchanged = []
    for name in ("get_autotune_config", "matmul_kernel"):
        nodes = [next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == name)
                 for tree in (original_ast, canonical_ast)]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
        unchanged.append(name)
    output.mkdir(parents=True, exist_ok=True)
    guards = admission()
    print("canonical guards admit witness; all 15 static assertions and grid=(1,) pass", flush=True)
    profiles = compile_profiles(output)
    exact = (-128) * (K // 4) * sum(((15 >> (2 * f)) & 3) - 1 for f in range(4))
    sync, wg = recurrence(K, True), recurrence(K, False)
    assert INT_MIN <= exact <= INT_MAX
    assert exact == -1073774592 and sync["output"] == -1073709056 and wg["output"] == exact
    assert sync["output"] - exact == 65536
    controls = []
    for k in (512, 2**24):
        target = -64 * k
        sat, wrap = recurrence(k, True), recurrence(k, False)
        assert sat["output"] == wrap["output"] == target
        controls.append({"K": k, "dense_result": target, "sync": sat, "wgmma": wrap})
    record = {
        "task": "int8_matmul_kernel", "exit_code": 0,
        "classification": "source_defect_pending_repair",
        "source": pin(SOURCE), "checker": pin(Path(__file__)),
        "original_source": pin(ORIGINAL), "unchanged_original_AST": unchanged,
        "witness": {"M": 1, "N": 1, "K": K, "A_FP32_value": -128,
                    "packed_B_uint8_value": 15, "unpacked_field_weights": [2, 2, -1, -1],
                    "A_strides": [K, 1], "B_strides": [1, 1],
                    "independent_dense_result": exact, "dense_result_fits_INT32": True,
                    "syncSat_config14": sync, "wrapping_config4": wg,
                    "difference": 65536},
        "admission": guards, "compiler_profiles": profiles, "arithmetic_controls": controls,
        "conclusion": "An admitted input with representable exact INT32 result yields a different result under canonical config14 carry-fed satfinite MMA; canonical config4 wrapping WGMMA agrees with the independent dense result. Autotuning can change an integer answer. This is an integer/source binding defect, not deferred FP numerics.",
        "preserved": "All prior source and proofs remain untouched. Existing unbounded mathematical and selected-ISA recurrence theorems remain valid within their declared interpretation; this record does not assert a GPU run or a new complete correctness proof.",
        "GPU_execution": False, "independent_numerical_replay": False,
        "evidence_kind": "Actual canonical guard execution with mocked CUDA, exact-source offline SM90 lowering, and exact integer instruction-recurrence calculation.",
    }
    receipt = output / "counterexample.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"exact_dense": exact, "sync_sat": sync["output"], "wgmma_wrap": wg["output"],
                      "receipt": str(receipt), "sha256": pin(receipt)["sha256"]}), flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".veritile-cache/kernel-evidence/int8_matmul_kernel/canonical-launch")
    args = parser.parse_args()
    run(args.output)
