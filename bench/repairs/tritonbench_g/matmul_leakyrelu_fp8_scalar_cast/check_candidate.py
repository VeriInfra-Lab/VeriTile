"""Check the nine-call source delta and compile the actual specialized ABI."""
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / "bench/repairs/tritonbench_g/matmul_leakyrelu_fp8/matmul_leakyrelu_fp8_repaired.py"
SOURCE = HERE / "matmul_leakyrelu_fp8_scalar_cast.py"
OUT = ROOT / ".veritile-cache/kernel-evidence/matmul_leakyrelu_fp8_scalar_cast"
SCALARS = ["M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_cm", "stride_cn"]


def pin(p):
    return {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.pop("TRITON_INTERPRET", None)
    os.environ["TRITON_CACHE_DIR"] = str(OUT / "compiler-cache")
    import torch
    import triton
    from torch._subclasses.fake_tensor import FakeTensorMode
    from triton._C.libtriton import native_specialize_impl
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource
    warnings.filterwarnings("ignore", message="Accessing the data pointer")
    expected = OLD.read_text()
    for name in SCALARS:
        before, after = f"{name} = {name}.to(tl.int64)", f"{name} = tl.cast({name}, tl.int64)"
        assert expected.count(before) == 1
        expected = expected.replace(before, after)
    assert SOURCE.read_text() == expected
    module = load(SOURCE, "fp8_scalar_cast_candidate")
    kernel = module.matmul_kernel.fn
    config, = module.matmul_kernel.configs
    assert config.kwargs == {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256,
                             "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}
    assert config.num_stages == 2 and config.num_warps == 8
    source_node = next(n for n in ast.parse(SOURCE.read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == "matmul_kernel")
    actual = ast.parse(kernel.src).body[0]
    assert ast.dump(actual.args) == ast.dump(source_node.args)
    assert [ast.dump(n) for n in actual.body] == [ast.dump(n) for n in source_node.body]
    target = GPUTarget("cuda", 90, 32)
    backend = CUDABackend(target)
    cases = [(m, m, 64, act) for m in (1, 2) for act in ("", "leaky_relu")]
    cases += [(129, 257, 65, ""), (129, 257, 65, "leaky_relu"),
              (1025, 257, 1, "leaky_relu"), (1, 3, 63, "")]
    records = []
    for index, (m, n, k, activation) in enumerate(cases):
        captured = {}

        @contextlib.contextmanager
        def device(owner):
            assert owner == torch.device("cuda:0")
            yield

        class Launch:
            def __getitem__(self, grid):
                def invoke(*args, **kwargs):
                    captured.update(args=args, kwargs=kwargs, grid=list(grid(config.kwargs)))
                return invoke

        with FakeTensorMode():
            a = torch.empty((m, k), dtype=torch.float32, device="cuda:0")
            b = torch.empty((k, n), dtype=torch.float32, device="cuda:0")
            with patch.object(module, "matmul_kernel", Launch()), patch.object(torch.cuda, "device", device):
                c = module.matmul(a, b, activation)
            assert c is captured["args"][2] and c.dtype == torch.float16
            assert list(captured["args"][3:]) == [m, n, k, k, 1, n, 1, n, 1]
            assert captured["grid"] == [((m+127)//128)*((n+255)//256)]
        pointers = [torch.empty((m, k), dtype=torch.float32), torch.empty((k, n), dtype=torch.float32),
                    torch.empty((m, n), dtype=torch.float16)]
        assert all(x.data_ptr() % 16 == 0 for x in pointers)
        arguments = dict(zip(kernel.arg_names[:12], pointers+list(captured["args"][3:])))
        arguments.update(captured["kwargs"], **config.kwargs)
        bound = {name: arguments[name] for name in kernel.arg_names}
        specialization = [("constexpr", bound[p.name]) if p.is_constexpr else
            native_specialize_impl(backend, bound[p.name], p.is_const, not p.do_not_specialize,
                                   not p.do_not_specialize_on_alignment) for p in kernel.params]
        options = {"num_warps": config.num_warps, "num_stages": config.num_stages, "num_ctas": config.num_ctas}
        _, signature, constants, attrs = kernel._pack_args(backend, options, bound, specialization, None)
        assert list(bound) == kernel.arg_names
        for i, name in enumerate(kernel.arg_names):
            if signature[name] == "constexpr":
                assert constants[(i,)] == bound[name], (name, constants[(i,)], bound[name])
        assert signature["stride_ak"] == "constexpr" and constants[(7,)] == 1
        compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                                  target=target, options=options)
        artifacts = []
        for ext in ("ttir", "ttgir", "llir", "ptx"):
            path = OUT / f"candidate_{index:02d}.{ext}"
            path.write_text(compiled.asm[ext]); artifacts.append(pin(path))
        assert "tt.dot" in compiled.asm["ttir"] and "tf32" in compiled.asm["ttir"]
        records.append({"shape": [m, n, k], "activation": activation, "grid": captured["grid"],
            "actual_wrapper_admitted": True, "actual_specialization_compile_success": True,
            "signature": signature, "constants": {kernel.arg_names[key[0]]: value for key, value in constants.items()},
            "attrs": {kernel.arg_names[key[0]]: value for key, value in attrs.items()},
            "options": options, "shared_memory_bytes": compiled.metadata.shared, "artifacts": artifacts})
        print(f"PASS actual specialized ABI {index}: M={m}, N={n}, K={k}, activation={activation!r}", flush=True)
    # The exact wrapper is unchanged; these cases preserve its pre-kernel exits.
    empty_cases = []
    with FakeTensorMode():
        for m, n, k in [(0, 3, 64), (3, 0, 64), (3, 5, 0)]:
            a = torch.empty((m, k), dtype=torch.float32, device="cuda:0")
            b = torch.empty((k, n), dtype=torch.float32, device="cuda:0")
            class NoLaunch:
                def __getitem__(self, grid):
                    raise AssertionError("empty/K0 branch submitted a kernel")
            with patch.object(module, "matmul_kernel", NoLaunch()):
                c = module.matmul(a, b, "leaky_relu")
            assert c.shape == (m, n) and c.dtype == torch.float16
            empty_cases.append([m, n, k])
    report = {"candidate": pin(SOURCE), "canonical_unchanged": pin(OLD), "checker": pin(Path(__file__)),
        "exact_source_delta": "Only nine runtime scalar/stride tensor-method casts become tl.cast; all other bytes identical.",
        "casts": SCALARS, "actual_specialized_compiles": records, "empty_K0_no_launch_cases": empty_cases,
        "target": "CUDA SM90 warp32", "triton_version": triton.__version__, "torch_version": torch.__version__,
        "GPU_execution": False, "primitive_numerical_experiment": False,
        "scope": "Actual canonical/candidate wrapper metadata and native binder ABI; matching aligned CPU pointer handles for offline dtype/alignment ABI only. No specialization disabled and no forced-dynamic ABI used."}
    path = OUT / "candidate-controls.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"actual_specialized_compiles": len(records), "empty_K0_controls": len(empty_cases), "receipt": pin(path)}), flush=True)


if __name__ == "__main__":
    run()
