"""Actual JIT specialization counterexample; no GPU or numerical experiment.

Import the unchanged canonical module, admit real Torch CUDA FakeTensor
metadata through its exact wrapper, capture its scalar arguments, and use
Triton's native argument specializer before exact-source offline compilation.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import traceback
import warnings
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "matmul_leakyrelu_fp8_repaired.py"
EXPECTED = "a5a6166c7e57e82e104805f7060e09a472a62d7be35e04ca7a9296bc5b87631e"


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def import_canonical():
    spec = importlib.util.spec_from_file_location("matmul_fp8_kernel_counterexample", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module.matmul_kernel.fn
    configs = module.matmul_kernel.configs
    assert len(configs) == 1
    assert configs[0].kwargs == {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256,
                                "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}
    assert configs[0].num_stages == 2 and configs[0].num_warps == 8
    source_node = next(n for n in ast.parse(SOURCE.read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == "matmul_kernel")
    jit_node = ast.parse(kernel.src).body[0]
    assert ast.dump(source_node.args) == ast.dump(jit_node.args)
    assert [ast.dump(n) for n in source_node.body] == [ast.dump(n) for n in jit_node.body]
    return module, kernel, configs[0]


def admitted_call(module, config, m, n, k, activation):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    captured, events = {}, []

    @contextlib.contextmanager
    def device(owner):
        assert owner == torch.device("cuda:0")
        events.append("device-enter")
        yield
        events.append("device-exit")

    class Launch:
        def __getitem__(self, grid):
            captured["grid"] = list(grid(config.kwargs))

            def call(*args, **kwargs):
                events.append("launch")
                captured.update(args=args, kwargs=kwargs)

            return call

    with FakeTensorMode():
        a = torch.empty((m, k), device="cuda:0", dtype=torch.float32)
        b = torch.empty((k, n), device="cuda:0", dtype=torch.float32)
        with patch.object(module, "matmul_kernel", Launch()), patch.object(torch.cuda, "device", device):
            c = module.matmul(a, b, activation)
        assert events == ["device-enter", "launch", "device-exit"]
        assert c is captured["args"][2]
        assert c.shape == (m, n) and c.dtype == torch.float16
        assert captured["grid"] == [1]
        assert list(captured["args"][3:]) == [m, n, k, k, 1, n, 1, n, 1]
        metadata = [{"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype),
                     "device": str(x.device), "contiguous": x.is_contiguous(), "numel": x.numel()}
                    for x in (a, b, c)]

    # Only pointer handles are replaced for the offline binder. These ordinary
    # CPU allocations have the same dtype and aligned-pointer specialization;
    # captured wrapper scalar values and all constexpr parameters are exact.
    # No CPU pointer is submitted to a device launcher.
    pointers = [torch.ones((m, k), dtype=torch.float32), torch.ones((k, n), dtype=torch.float32),
                torch.empty((m, n), dtype=torch.float16)]
    assert all(x.data_ptr() % 16 == 0 for x in pointers)
    values = pointers + list(captured["args"][3:])
    witness = {"shape": {"M": m, "N": n, "K": k}, "activation": activation,
               "actual_canonical_wrapper_admitted": True, "Torch_FakeTensor_metadata": metadata,
               "captured_scalar_arguments": values[3:], "grid": captured["grid"], "events": events,
               "input_values": "A and B all +1; the physical CUDA instance is a mathematical witness, not a GPU allocation in this run",
               "expected_each_output_real": k,
               "expected_each_output_FP16_bits": hex(struct.unpack("<H", struct.pack("<e", float(k)))[0]),
               "boundary": "Actual Torch CUDA FakeTensor wrapper admission and exact scalar capture; aligned CPU pointer handles supply the identical offline pointer ABI; no GPU launch."}
    return values, captured["kwargs"], witness


def specialize(kernel, values, keywords, config, backend):
    from triton._C.libtriton import native_specialize_impl
    bound = dict(zip(kernel.arg_names[:12], values))
    bound.update(keywords, **config.kwargs)
    assert set(bound) == set(kernel.arg_names)
    # _pack_args consumes bound.values() in parameter order, as the generated
    # runtime binder does. Keyword insertion order is not parameter order.
    bound = {name: bound[name] for name in kernel.arg_names}
    specialization = [("constexpr", bound[p.name]) if p.is_constexpr else
                      native_specialize_impl(backend, bound[p.name], p.is_const,
                                             not p.do_not_specialize,
                                             not p.do_not_specialize_on_alignment)
                      for p in kernel.params]
    options = {"num_warps": config.num_warps, "num_stages": config.num_stages,
               "num_ctas": config.num_ctas}
    _, signature, constants, attrs = kernel._pack_args(backend, options, bound, specialization, None)
    assert list(bound) == kernel.arg_names
    for index, name in enumerate(kernel.arg_names):
        if signature[name] == "constexpr":
            assert constants[(index,)] == bound[name], (name, constants[(index,)], bound[name])
    for name, value in config.kwargs.items():
        assert constants[(kernel.arg_names.index(name),)] == value
    assert constants[(kernel.arg_names.index("ACTIVATION"),)] == keywords["ACTIVATION"]
    assert all(signature[name] == "constexpr" for name in ("stride_ak", "stride_bn", "stride_cn"))
    assert constants[(kernel.arg_names.index("stride_ak"),)] == 1
    return signature, constants, attrs, options


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    os.environ.pop("TRITON_INTERPRET", None)
    os.environ["TRITON_CACHE_DIR"] = str(output / "compiler-cache")
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource
    from triton.compiler.errors import CompilationError
    warnings.filterwarnings("ignore", message="Accessing the data pointer")
    assert pin(SOURCE)["sha256"] == EXPECTED
    module, kernel, config = import_canonical()
    target = GPUTarget("cuda", 90, 32)
    backend = CUDABackend(target)
    failures, witnesses = [], []
    for m, n, expression in [(1, 1, "M = M.to(tl.int64)"),
                              (2, 2, "stride_ak = stride_ak.to(tl.int64)")]:
        for activation in ("", "leaky_relu"):
            values, keywords, witness = admitted_call(module, config, m, n, 64, activation)
            witnesses.append(witness)
            signature, constants, attrs, options = specialize(kernel, values, keywords, config, backend)
            if m == 1:
                assert signature["M"] == "constexpr" and constants[(3,)] == 1
            else:
                assert signature["M"] == "i32" and signature["N"] == "i32"
            print(f"admitted M={m}, N={n}, K=64, activation={activation!r}; native stride_ak=constexpr(1)", flush=True)
            try:
                triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                               target=target, options=options)
            except CompilationError as error:
                message = str(error)
                assert expression in message and "'int' object has no attribute 'to'" in message, message
                path = output / f"M{m}_N{n}_{'active' if activation else 'identity'}.error.txt"
                path.write_text(traceback.format_exc())
                failures.append({"shape": witness["shape"], "activation": activation,
                                 "signature": signature,
                                 "constants": {kernel.arg_names[k[0]]: v for k, v in constants.items()},
                                 "attrs": {kernel.arg_names[k[0]]: v for k, v in attrs.items()},
                                 "options": options, "failure_site": expression,
                                 "error_type": type(error).__name__, "error": message, "trace": pin(path)})
                print(f"confirmed CompilationError: {expression}", flush=True)
            else:
                raise AssertionError("Canonical source unexpectedly compiled under actual specialization")

    # Diagnostic contrast only: suppress the runtime binder's scalar-one
    # specialization in ASTSource. This is intentionally NOT the wrapper ABI.
    # It explains why all-dynamic offline profiles can miss this source defect.
    dynamic = {name: ("*fp32" if name in ("a_ptr", "b_ptr") else "*fp16" if name == "c_ptr"
                      else "constexpr" if p.is_constexpr else "i32")
               for name, p in zip(kernel.arg_names, kernel.params)}
    dynamic_constants = {(p.num,): values for p in kernel.params if p.is_constexpr
                         for values in [keywords[p.name] if p.name in keywords else config.kwargs[p.name]]}
    compiled = triton.compile(ASTSource(kernel, signature=dynamic, constexprs=dynamic_constants, attrs={}),
                              target=target, options=options)
    artifacts = []
    for ext in ("ttir", "ttgir", "llir", "ptx"):
        path = output / f"unchanged_source_forced_dynamic_control.{ext}"
        path.write_text(compiled.asm[ext])
        artifacts.append(pin(path))
    assert "tt.dot" in compiled.asm["ttir"] and "tf32" in compiled.asm["ttir"]
    print("unchanged-source forced-dynamic diagnostic control compiled; this is not the wrapper ABI", flush=True)
    record = {"task": "matmul_leakyrelu_fp8", "diagnostic_exit_code": 0,
              "classification": "canonical_checked_source_defect_pending_repair",
              "canonical_source": pin(SOURCE), "checker": pin(Path(__file__)),
              "triton_version": triton.__version__, "torch_version": torch.__version__,
              "target": "CUDA SM90, warp32", "config": config.kwargs, "options": options,
              "kernel_body_and_signature_match_canonical": True,
              "JIT_binding": "Actual imported Autotuner.fn with unchanged @triton.jit; native_specialize_impl and JITFunction._pack_args applied to captured wrapper scalars and matching aligned pointer ABI.",
              "witnesses": witnesses, "compile_failures": failures,
              "forced_dynamic_control": {"source_unchanged": True, "signature": dynamic,
                  "constants": {kernel.arg_names[k[0]]: v for k, v in dynamic_constants.items()},
                  "status": "offline_compile_success", "shared_memory_bytes": compiled.metadata.shared,
                  "artifacts": artifacts,
                  "scope": "Manually forced dynamic scalar ABI, intentionally different from actual native specialization; diagnostic only, no canonical kernel-success claim."},
              "canonical_source_changed": False, "new_admission_guards": [],
              "numerical_assumptions": [], "deferred_experiments": [],
              "GPU_execution": False, "primitive_numerical_experiment": False,
              "conclusion": "Legal positive work fails before producing output: M=1 is a constexpr integer at M.to; even M=N=2 reaches the obligatory contiguous stride_ak=1 as a constexpr integer at stride_ak.to. Existing abstract source/proof results remain preserved within their model boundaries."}
    path = output / "counterexample.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"failures": len(failures), "forced_dynamic_controls": 1, "receipt": pin(path)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".veritile-cache/kernel-evidence/matmul_leakyrelu_fp8/constexpr-cast")
    run(parser.parse_args().output)
