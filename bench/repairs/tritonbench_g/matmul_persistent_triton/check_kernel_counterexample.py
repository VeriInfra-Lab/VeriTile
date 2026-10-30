"""Exact-source persistent matmul viability under ordinary JIT specialization.

Both explicit source dtype/configuration routes are retained. Actual wrapper
arguments are captured with real Torch CUDA FakeTensor metadata; offline
native specialization is ordered and checked against kernel.arg_names.
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
import types
import warnings
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "matmul_persistent_triton_repaired.py"
EXPECTED = "ad6eb70f7493aff7f323558870765887e09c208c3a547e2c6271fe519efddd7d"
KERNEL = "matmul_kernel_persistent"


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def import_canonical():
    spec = importlib.util.spec_from_file_location("persistent_kernel_counterexample", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, KERNEL)
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == KERNEL)
    actual = ast.parse(kernel.src).body[0]
    assert ast.dump(node.args) == ast.dump(actual.args)
    assert [ast.dump(n) for n in node.body] == [ast.dump(n) for n in actual.body]
    return module, kernel


def admitted_call(module, dtype, m, n, k):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    captured, events = {}, []

    def properties(owner):
        assert owner == torch.device("cuda:0")
        events.append("owning-device-properties")
        return types.SimpleNamespace(multi_processor_count=132)

    @contextlib.contextmanager
    def device(owner):
        assert owner == torch.device("cuda:0")
        events.append("device-enter")
        yield
        events.append("device-exit")

    class Launch:
        def __getitem__(self, grid):
            def invoke(*args, **kwargs):
                events.append("launch")
                captured.update(args=args, kwargs=kwargs, grid=list(grid(kwargs)))
            return invoke

    with FakeTensorMode():
        a = torch.empty((m, k), device="cuda:0", dtype=dtype)
        b = torch.empty((k, n), device="cuda:0", dtype=dtype)
        with patch.object(module, KERNEL, Launch()), \
                patch.object(torch.cuda, "get_device_properties", properties), \
                patch.object(torch.cuda, "device", device):
            c = module.matmul_persistent(a, b)
        assert c is captured["args"][2]
        assert c.shape == (m, n) and c.dtype == dtype
        assert captured["grid"] == [1]
        assert events == ["owning-device-properties", "device-enter", "launch", "device-exit"]
        assert list(captured["args"][3:]) == [m, n, k, k, 1, n, 1, n, 1]
        metadata = [{"shape": list(x.shape), "strides": list(x.stride()), "dtype": str(x.dtype),
                     "device": str(x.device), "numel": x.numel(), "element_bytes": x.element_size(),
                     "contiguous": x.is_contiguous()} for x in (a, b, c)]
    # Matching dtype and alignment ABI only. These CPU pointers never go to a
    # CUDA launcher; the error depends on the exact captured scalar arguments.
    pointers = [torch.ones((m, k), dtype=torch.float32).to(dtype),
                torch.ones((k, n), dtype=torch.float32).to(dtype), torch.empty((m, n), dtype=dtype)]
    assert all(p.data_ptr() % 16 == 0 for p in pointers)
    values = pointers + list(captured["args"][3:])
    half = dtype == torch.float16
    assert captured["kwargs"] == {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64 if half else 128,
        "GROUP_SIZE_M": 8, "NUM_SMS": 132, "num_stages": 3 if half else 4, "num_warps": 8}
    witness = {"dtype": str(dtype), "shape": {"M": m, "N": n, "K": k},
        "actual_canonical_wrapper_admitted": True, "metadata": metadata,
        "captured_scalar_arguments": values[3:], "captured_keywords": captured["kwargs"],
        "grid": captured["grid"], "events": events, "input_values": "all +1",
        "expected_each_output_real": k,
        "expected_each_output_bits": hex(struct.unpack("<H", struct.pack("<e", float(k)))[0]) if half else "0x70",
        "boundary": "Actual Torch CUDA FakeTensor wrapper admission; ordinary device property NUM_SMS=132; captured scalars/config; matching aligned CPU pointer handles for offline ABI only. No physical CUDA allocation or execution."}
    return values, captured["kwargs"], witness


def specialize(kernel, values, keywords, backend):
    from triton._C.libtriton import native_specialize_impl
    arguments = dict(zip(kernel.arg_names[:12], values))
    arguments.update({name: value for name, value in keywords.items() if name in kernel.arg_names})
    assert set(arguments) == set(kernel.arg_names)
    bound = {name: arguments[name] for name in kernel.arg_names}
    specialization = [("constexpr", bound[p.name]) if p.is_constexpr else
        native_specialize_impl(backend, bound[p.name], p.is_const, not p.do_not_specialize,
                               not p.do_not_specialize_on_alignment) for p in kernel.params]
    options = {"num_stages": keywords["num_stages"], "num_warps": keywords["num_warps"]}
    _, signature, constants, attrs = kernel._pack_args(backend, options, bound, specialization, None)
    assert list(bound) == kernel.arg_names
    for i, name in enumerate(kernel.arg_names):
        if signature[name] == "constexpr":
            assert constants[(i,)] == bound[name], (name, constants[(i,)], bound[name])
    for name in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_SMS"):
        assert constants[(kernel.arg_names.index(name),)] == keywords[name]
    assert signature["stride_ak"] == "constexpr" and constants[(7,)] == 1
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
    module, kernel = import_canonical()
    target = GPUTarget("cuda", 90, 32)
    backend = CUDABackend(target)
    witnesses, failures, controls = [], [], []
    for profile, dtype, k, pointer_type in [("fp16", torch.float16, 64, "*fp16"),
                                          ("e4m3", torch.float8_e4m3fn, 128, "*fp8e4nv")]:
        for m, n, expression in [(1, 1, "M.to(tl.int64)"), (2, 2, "stride_ak.to(tl.int64)")]:
            values, keywords, witness = admitted_call(module, dtype, m, n, k)
            witnesses.append(witness)
            signature, constants, attrs, options = specialize(kernel, values, keywords, backend)
            assert signature["a_ptr"] == signature["b_ptr"] == signature["c_ptr"] == pointer_type
            if m == 1:
                assert signature["M"] == "constexpr" and constants[(3,)] == 1
            else:
                assert signature["M"] == signature["N"] == "i32"
            print(f"admitted {profile} M={m}, N={n}, K={k}; exact config and all constexpr values checked", flush=True)
            try:
                triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                               target=target, options=options)
            except CompilationError as error:
                message = str(error)
                assert expression in message and "'int' object has no attribute 'to'" in message, message
                trace = output / f"{profile}_M{m}_N{n}.error.txt"
                trace.write_text(traceback.format_exc())
                failures.append({"profile": profile, "shape": witness["shape"], "signature": signature,
                    "constants": {kernel.arg_names[key[0]]: value for key, value in constants.items()},
                    "attrs": {kernel.arg_names[key[0]]: value for key, value in attrs.items()},
                    "options": options, "failure_site": expression, "error_type": type(error).__name__,
                    "error": message, "trace": pin(trace)})
                print(f"confirmed {profile} CompilationError at {expression}", flush=True)
            else:
                raise AssertionError("canonical source unexpectedly compiled under actual specialization")

        # Same source, deliberately different all-dynamic scalar ABI. This
        # compiler control is not a successful canonical wrapper invocation.
        dynamic = {p.name: pointer_type if p.name in ("a_ptr", "b_ptr", "c_ptr")
                   else "constexpr" if p.is_constexpr else "i32" for p in kernel.params}
        dynamic_constants = {(i,): keywords[p.name] for i, p in enumerate(kernel.params) if p.is_constexpr}
        compiled = triton.compile(ASTSource(kernel, signature=dynamic, constexprs=dynamic_constants, attrs={}),
                                  target=target, options=options)
        artifacts = []
        for ext in ("ttir", "ttgir", "llir", "ptx"):
            path = output / f"{profile}_unchanged_source_forced_dynamic_control.{ext}"
            path.write_text(compiled.asm[ext]); artifacts.append(pin(path))
        assert "tt.dot" in compiled.asm["ttir"]
        controls.append({"profile": profile, "source_unchanged": True, "signature": dynamic,
            "constants": {kernel.arg_names[key[0]]: value for key, value in dynamic_constants.items()},
            "options": options, "status": "offline_compile_success", "shared_memory_bytes": compiled.metadata.shared,
            "artifacts": artifacts, "scope": "Intentionally forced-dynamic scalar ABI, not the canonical wrapper ABI; no GPU execution or kernel-correctness acceptance."})
        print(f"{profile} same-source forced-dynamic control compiled, shared={compiled.metadata.shared}; diagnostic ABI only", flush=True)
    record = {"task": "matmul_persistent_triton", "diagnostic_exit_code": 0,
        "classification": "canonical_checked_source_defect_pending_repair", "canonical_source": pin(SOURCE),
        "checker": pin(Path(__file__)), "triton_version": triton.__version__, "torch_version": torch.__version__,
        "target": "CUDA SM90, warp32", "kernel_body_and_signature_match_canonical": True,
        "JIT_binding": "Actual canonical JIT object; exact wrapper scalar/config capture; native_specialize_impl and parameter-ordered _pack_args; every constexpr and every pointer dtype checked.",
        "witnesses": witnesses, "compile_failures": failures, "forced_dynamic_controls": controls,
        "source_changed": False, "new_admission_guards": [], "numerical_assumptions": [],
        "deferred_experiments": [], "GPU_execution": False, "primitive_numerical_experiment": False,
        "conclusion": "Both retained dtype/configuration routes admit positive calls that fail at tensor .to on a constexpr integer-one scalar. No output is produced. Prior abstract proof results remain preserved; forced-dynamic compiler controls do not establish the real wrapper ABI."}
    path = output / "counterexample.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"canonical_failures": len(failures), "forced_dynamic_controls": len(controls), "receipt": pin(path)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".veritile-cache/kernel-evidence/matmul_persistent_triton/constexpr-cast")
    run(parser.parse_args().output)
