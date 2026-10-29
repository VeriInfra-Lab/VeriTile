"""Two directed native-binder compiler profiles for the independent word-add candidate.

Offline SM90 compilation only. CPU tensors select pointer dtype/alignment;
neither those tensors nor a GPU kernel are executed. Historical defect controls
are preserved, not rerun. Run with --compile once, then inspect saved artifacts
without that option. This is a compiler-profile check, not whole-kernel proof.
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
SOURCE = HERE / "int8_matmul_kernel_modular_checked.py"
PREVIOUS = ROOT / "bench/repairs/tritonbench_g/int8_matmul_kernel/int8_matmul_kernel_repaired.py"
PREVIOUS_SHA = "e95569108e080703b0bc8df8a61c862eba863cefe6fa4c6321c9ced9ca496cee"
SOURCE_SHA = "5fb5ed4e23f10e9e65aba367755153d8562ef7812ca8c64ee753ca9569af5059"
K = 2**24 + 512


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source_delta():
    assert pin(PREVIOUS)["sha256"] == PREVIOUS_SHA
    assert pin(SOURCE)["sha256"] == SOURCE_SHA
    old, new = [ast.parse(p.read_text()) for p in (PREVIOUS, SOURCE)]
    expected = ast.parse('''partial = tl.dot(a, (b - tensor_full), out_dtype=tl.int32)
accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;", constraints="=r,r,r", args=[accumulator, partial],
    dtype=tl.int32, is_pure=True, pack=1)
''').body
    replacements = 0
    for node in ast.walk(new):
        if not isinstance(node, ast.For):
            continue
        for index in range(len(node.body)-1):
            if all(ast.dump(node.body[index+j]) == ast.dump(expected[j]) for j in range(2)):
                node.body[index:index+2] = ast.parse(
                    "accumulator += tl.dot(a, (b - tensor_full), out_dtype=tl.int32)").body
                replacements += 1
                break
    assert replacements == 1
    wrappers = [node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == "matmul"]
    assert ast.get_docstring(wrappers[-1]) == "Checked FP32-by-packed-uint8 contract with signed low32 output."
    wrappers[-1].body[0].value.value = "Checked FP32-by-packed-uint8 contract for the unchanged autotuned source."
    assert ast.dump(old) == ast.dump(new), "Unexpected source AST change outside the exact arithmetic delta and docstring"
    return {"old_source": pin(PREVIOUS), "candidate_source": pin(SOURCE),
            "exact_arithmetic_replacement_count": replacements,
            "all_other_AST_unchanged_except_contract_docstring": True,
            "all_15_configurations_and_wrapper_guards_preserved": True}


def compile_profiles(output):
    driver_snapshot = output / "compiler-driver-at-compilation.py"
    assert not driver_snapshot.exists(), "Preserve the compiler driver used by an earlier run"
    driver_snapshot.write_bytes(Path(__file__).read_bytes())
    os.environ.pop("TRITON_INTERPRET", None)
    os.environ["TRITON_CACHE_DIR"] = str(output / "compiler-cache")
    os.environ["TRITON_DUMP_DIR"] = str(output / "compiler-dump")
    os.environ["TRITON_OVERRIDE_DIR"] = str(output / "compiler-override")
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.runtime.jit import create_function_from_signature
    spec = importlib.util.spec_from_file_location("int8_modular_candidate", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module.matmul_kernel.fn
    configs = module.get_autotune_config()
    assert len(configs) == 15
    target = GPUTarget("cuda", 90, 32)
    backend = CUDABackend(target)
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    tokens = [torch.empty(1, dtype=dtype) for dtype in (torch.float32, torch.uint8, torch.int32)]
    assert all(token.data_ptr() % 16 == 0 for token in tokens)
    profiles = []
    for index in (14, 4):
        config = configs[index]
        arguments = [*tokens, 1, 1, K, K, 1, 1, 1, 1, 1]
        launch_options = {"num_warps": config.num_warps, "num_stages": config.num_stages,
                          "num_ctas": config.num_ctas}
        bound, specialization, binder_options = binder(*arguments, **config.kwargs, **launch_options)
        parsed, signature, constants, attrs = kernel._pack_args(
            backend, binder_options, bound, specialization, binder_options)
        assert signature["M"] == signature["N"] == "constexpr"
        assert [signature[name] for name in ("a_ptr", "b_ptr", "c_ptr")] == ["*fp32", "*u8", "*i32"]
        assert list(bound) == kernel.arg_names
        compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                                  target=target, options=launch_options)
        artifacts = []
        for extension in ("ttir", "ttgir", "llir", "ptx"):
            path = output / f"config{index}.{extension}"
            path.write_text(compiled.asm[extension])
            artifacts.append(pin(path))
        profiles.append({
            "index": index, "configuration": config.kwargs,
            "binding": "Actual create_function_from_signature binder and JITFunction._pack_args used by JITFunction.run; explicit offline CUDA backend.",
            "scalar_arguments": {name: value for name, value in bound.items() if name not in ("a_ptr", "b_ptr", "c_ptr")},
            "pointer_tokens": [{"dtype": str(t.dtype), "alignment_mod16": t.data_ptr() % 16} for t in tokens],
            "signature": signature,
            "constants": {kernel.arg_names[key[0]]: value for key, value in constants.items()},
            "attrs": {kernel.arg_names[key[0]]: value for key, value in attrs.items()},
            "options": launch_options, "parsed_options": parsed.__dict__,
            "target": "cuda sm90 warp32", "triton_version": triton.__version__,
            "artifacts": artifacts,
        })
        print(f"compiled candidate config {index} with actual binder", flush=True)
    record = {"source": pin(SOURCE), "compiler_driver": pin(Path(__file__)), "profiles": profiles,
              "GPU_execution": False, "full_witness_contents_allocated": False,
              "whole_kernel_lowering_correspondence_assumed": False}
    (output / "binder-profiles.json").write_text(json.dumps(record, indent=2) + "\n")


def inspect_profile(output, profile):
    index = profile["index"]
    for ref in profile["artifacts"]:
        assert pin(ROOT / ref["path"])["sha256"] == ref["sha256"]
    ttir, ttgir, llir, ptx = [(output / f"config{index}.{ext}").read_text()
                             for ext in ("ttir", "ttgir", "llir", "ptx")]
    dots = [line.strip() for line in ttir.splitlines() if "= tt.dot " in line]
    assert dots
    zero_constants = set(re.findall(r"(%[\w]+) = arith.constant dense<0> : tensor<[^\n]+xi32>", ttir))
    seeds = []
    for line in dots:
        match = re.search(r"(%\w+) = tt.dot %\w+, %\w+, (%\w+)", line)
        assert match and match.group(2) in zero_constants, line
        seeds.append({"result": match.group(1), "seed": match.group(2), "line": line})
    assembly = [line.strip() for line in ttir.splitlines()
                if '= tt.elementwise_inline_asm "add.u32 $0, $1, $2;"' in line]
    assert len(assembly) == len(dots) == 1, (dots, assembly)
    assert '"=r,r,r"' in assembly[0] and "packed_element = 1 : i32" in assembly[0]
    assert "pure = true" in assembly[0]
    asm_match = re.search(r'(%\w+) = tt.elementwise_inline_asm .*\} (%\w+), (%\w+) :', assembly[0])
    assert asm_match and asm_match.group(3) == seeds[0]["result"]
    output_name, carry_name, partial_name = asm_match.groups()
    assert any(carry_name + " = " in line for line in ttir.splitlines() if "scf.for" in line)
    assert any(output_name in line for line in ttir.splitlines() if "scf.yield" in line)
    gpu_dots = [line.strip() for line in ttgir.splitlines()
                if "= tt.dot " in line or "= ttng.warp_group_dot " in line]
    gpu_zero_constants = set(re.findall(r"(%\w+) = arith.constant dense<0> : tensor<[^\n]+xi32,", ttgir))
    assert len(gpu_dots) == 1
    gpu_seed = re.search(r'(?:tt\.dot|ttng\.warp_group_dot) %\w+, %\w+, (%\w+)', gpu_dots[0])
    assert gpu_seed and gpu_seed.group(1) in gpu_zero_constants
    assert 'tt.elementwise_inline_asm "add.u32 $0, $1, $2;"' in ttgir
    assert re.search(r"arith\.fptosi .*xf32> to tensor<.*xi8>", ttir)
    assert re.search(r"fptosi <\d+ x float> .* to <\d+ x i8>", llir)
    assert 'asm "add.u32 $0, $1, $2;"' in llir
    assert "add.u32" in ptx
    llvm_adds = re.findall(r'(%\w+) = tail call i32 asm "add.u32 \$0, \$1, \$2;", "=r,r,r"\(i32 (%\w+), i32 (%\w+)\)', llir)
    assert len(llvm_adds) == (8 if index == 14 else 64)
    for result, carry, partial in llvm_adds:
        phi = re.search(r'^\s*' + re.escape(carry) + r' = phi i32 ([^\n]+)', llir, re.MULTILINE)
        assert phi and re.search(r'\[ ' + re.escape(result) + r', ', phi.group(1)), (result, carry, partial)
    native_instructions = []
    ptx_zero_initializations = []
    if index == 14:
        family = "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"
        assert family in ptx and "wgmma.mma_async" not in ptx
        llvm_mma = [line.strip() for line in llir.splitlines() if 'asm sideeffect "' + family in line]
        assert len(llvm_mma) == 2
        assert all('"=r,=r,=r,=r,0,1,2,3,r,r,r,r,r,r"(i32 0, i32 0, i32 0, i32 0,' in line for line in llvm_mma)
        # Check the actual straight-line PTX seed definitions. Reset facts at
        # labels so an incoming control-flow path cannot supply a stale zero.
        zero_registers = set()
        for raw in ptx.splitlines():
            line = raw.strip()
            if line.endswith(":"):
                zero_registers.clear()
            move = re.fullmatch(r'mov\.b32\s+(%r\d+),\s*(%r\d+|0);', line)
            if move:
                dest, value = move.groups()
                if value == "0" or value in zero_registers:
                    zero_registers.add(dest)
                    ptx_zero_initializations.append(line)
                else:
                    zero_registers.discard(dest)
            elif line.startswith(family):
                groups = [re.findall(r'%r\d+', g) for g in re.findall(r'\{([^}]+)\}', line)]
                assert len(groups) == 4 and all(r in zero_registers for r in groups[3]), line
                native_instructions.append(line)
                zero_registers.difference_update(groups[0])
            else:
                # A generic scalar destination invalidates a known-zero fact.
                dest = re.match(r'[\w.]+\s+(%r\d+),', line)
                if dest:
                    zero_registers.discard(dest.group(1))
        assert len(native_instructions) == 2
    else:
        family = "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8"
        assert family in ptx
        instructions = [line.strip() for line in llir.splitlines() if 'asm sideeffect "wgmma.mma_async' in line]
        assert len(instructions) == 1
        assert all(', $68, 0;"' in line for line in instructions), instructions
        assert all(re.search(r'"\(i32 %\w+, i32 %\w+, i32 %\w+, i32 %\w+, i64 %\w+\)', line) for line in instructions)
        native_instructions = [line.strip() for line in ptx.splitlines() if line.strip().startswith(family)]
        assert len(native_instructions) == 1 and all(line.endswith(", 0;") for line in native_instructions)
    partial_registers = [reg for line in native_instructions
                         for reg in re.findall(r'%r\d+', re.findall(r'\{([^}]+)\}', line)[0])]
    ptx_adds = re.findall(r'^\s*add\.u32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+);', ptx, re.MULTILINE)
    assert [partial for dest, carry, partial in ptx_adds] == partial_registers
    assert all(dest == carry and dest not in partial_registers for dest, carry, partial in ptx_adds)
    return {"index": index, "ttir_dot_zero_seeds": seeds, "ttir_distinct_add": assembly,
            "ttir_carried_add_result": {"result": output_name, "carry": carry_name, "partial": partial_name},
            "ttgir_dot_zero_seed": gpu_dots,
            "LLVM_distinct_add_sites": len(llvm_adds), "PTX_distinct_add_sites": len(ptx_adds),
            "PTX_instructions": native_instructions,
            "PTX_zero_seed_initializations": ptx_zero_initializations if index == 14 else ["WGMMA scale-d literal 0"],
            "PTX_distinct_accumulator_and_partial_registers": True,
            "instruction_family": family, "FP32_to_signed8_conversion_preserved": True,
            "block_size_K": profile["configuration"]["BLOCK_SIZE_K"],
            "general_block_prefix_absolute_bound": 32768,
            "scope": "This saved scalar specialization/configuration; no GPU execution or whole-kernel lowering assumption."}


def run(output, compile_requested):
    delta = source_delta()
    output.mkdir(parents=True, exist_ok=True)
    if compile_requested:
        assert not (output / "binder-profiles.json").exists(), "Preserve prior compiler profiles; inspect without --compile"
        compile_profiles(output)
    record = json.loads((output / "binder-profiles.json").read_text())
    assert record["source"] == pin(SOURCE)
    driver_snapshot = output / "compiler-driver-at-compilation.py"
    assert pin(driver_snapshot)["sha256"] == record["compiler_driver"]["sha256"]
    assert [p["index"] for p in record["profiles"]] == [14, 4]
    checks = [inspect_profile(output, profile) for profile in record["profiles"]]
    assert source_delta() == delta
    result = {"task": "int8_matmul_kernel_modular_checked", "status": "DIRECTED_COMPILER_PROFILES_CHECKED",
              "source_delta": delta, "checker": pin(Path(__file__)),
              "binder_profiles": pin(output / "binder-profiles.json"), "checks": checks,
              "compiler_driver_at_compilation": pin(driver_snapshot),
              "compiler_driver_reference_scope": "binder-profiles.json pins the original executed checker bytes, preserved in compiler-driver-at-compilation.py; current checker only reinspects saved artifacts.",
              "initial_inspection_failure": (
                  "Both compiles succeeded. Initial checker expected the TTIR attribute spelling pack instead of packed_element; fixed without recompilation."
                  if record["compiler_driver"]["sha256"] == "ddbe398a4abac1ae292c190e91595a3d8684f122908afd9320ced4a7b350298e" else None),
              "GPU_execution": False, "whole_kernel_proof": False,
              "historical_controls_rerun": False, "shared_DSL_or_core_changed": False}
    path = output / "compiler-checks.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"receipt": pin(path), "configurations": [p["index"] for p in checks]}), flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".veritile-cache/kernel-evidence/int8_matmul_kernel_modular_checked/directed-compiler")
    args = parser.parse_args()
    run(args.output.resolve(), args.compile)
