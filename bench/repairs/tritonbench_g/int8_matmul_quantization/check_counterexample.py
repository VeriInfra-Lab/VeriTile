"""Check admitted-input constexpr `.to` failures in both pinned JIT kernels.

The assertions and launch arguments are the actual canonical AST. CPU tensors
supply real shapes, storage, dtypes and values; CUDA ownership, allocation and
launch are mocked. Native Triton specialization is used for offline compilation.
No GPU executes and no numerical tolerance is tested.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "int8_matmul_quantization_repaired.py"
ORIGINAL = ROOT / "bench/tritonbench_g/int8_matmul_quantization/int8_matmul_quantization.py"
SOURCE_SHA = "2023c3fc3c849aa9a93b45bb5f672dd0d4412d175a713978dde6f70c5b164a81"
NAMES = {"producer": "quantize_int8_perrow_kernel", "gemm": "matmul_kernel"}


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def function(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def load_kernel(path, kind, module_name):
    import torch
    import triton
    import triton.language as tl
    tree = ast.parse(path.read_text())
    node = copy.deepcopy(function(tree, NAMES[kind]))
    configs_expr = next(k.value for k in node.decorator_list[0].keywords if k.arg == "configs")
    # Omit only the host autotune decorator. Retain @triton.jit, the exact
    # signature, body, source filename and original line numbers.
    node.decorator_list = [node.decorator_list[-1]]
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, triton=triton, tl=tl)
    sys.modules[module_name] = module
    exec(compile(ast.Module([node], []), str(path), "exec"), module.__dict__)
    configs = eval(compile(ast.Expression(configs_expr), str(path), "eval"), module.__dict__)
    kernel = getattr(module, NAMES[kind])
    actual = ast.parse(kernel.src).body[0]
    assert [ast.dump(n) for n in actual.body] == [ast.dump(n) for n in node.body]
    assert ast.dump(actual.args) == ast.dump(node.args)
    return kernel, configs, tree


def admitted_call(tree, configs, kind, m, n=2, k=32):
    import torch
    import triton

    class Tensor:
        def __init__(self, data):
            self.data = data
            self.device = torch.device("cuda:0")
            self.is_cuda = True

        def __getattr__(self, name):
            return getattr(self.data, name)

    events, captured = [], {}

    @contextlib.contextmanager
    def device(owner):
        assert owner == torch.device("cuda:0")
        events.append("device-enter")
        yield
        events.append("device-exit")

    def allocate(factory, shape, device, dtype):
        assert device == torch.device("cuda:0")
        events.append("allocation")
        return Tensor(factory(shape, dtype=dtype))

    class Launch:
        def __getitem__(self, grid):
            captured["grid"] = grid

            def call(*args, **kwargs):
                events.append("launch")
                captured.update(args=args, kwargs=kwargs)

            return call

    fake = types.SimpleNamespace(
        Tensor=Tensor, strided=torch.strided, int8=torch.int8,
        float32=torch.float32, float16=torch.float16,
        empty=lambda shape, **kw: allocate(torch.empty, shape, **kw),
        zeros=lambda shape, **kw: allocate(torch.zeros, shape, **kw),
        cuda=types.SimpleNamespace(device=device),
        _C=types.SimpleNamespace(_overlaps=lambda a, b: torch._C._overlaps(a.data, b.data)),
    )
    namespace = {"torch": fake, "triton": triton, NAMES[kind]: Launch()}
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_GEMM_TILES" for t in node.targets)]
    for name in ("_tensor", "_check_row", "_check_gemm", "quantize_int8_perrow", "matmul_int8"):
        nodes.append(function(tree, name))
    exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), namespace)
    if kind == "producer":
        inputs = [Tensor(torch.ones((m, k), dtype=torch.float32))]
        result = namespace["quantize_int8_perrow"](*inputs)
        assert result == (captured["args"][1], captured["args"][2])
        assert events == ["allocation", "allocation", "device-enter", "launch", "device-exit"]
        assert captured["grid"] == (m,)
        assert len(configs) == 4
        strides = list(captured["args"][5:10])
        assert strides == [k, 1, k, 1, 1]
        grid_records = [[m]] * len(configs)
        contents = {"FP32_input": "all +1", "Real_target_scale": "1/127", "Real_target_code": 127}
    else:
        inputs = [Tensor(torch.ones(shape, dtype=dtype)) for shape, dtype in
                  [((m, k), torch.int8), ((m,), torch.float16),
                   ((k, n), torch.int8), ((n,), torch.float32)]]
        result = namespace["matmul_int8"](*inputs)
        assert result is captured["args"][4]
        assert events == ["allocation", "device-enter", "launch", "device-exit"]
        assert len(configs) == 34
        strides = list(captured["args"][8:16])
        assert strides == [k, 1, 1, n, 1, 1, n, 1]
        grid_records = [list(captured["grid"](cfg.kwargs)) for cfg in configs]
        assert all(grid == [1, cfg.kwargs["SPLIT_K"]] for grid, cfg in zip(grid_records, configs))
        contents = {"INT8_A": "all +1", "INT8_B": "all +1",
                    "FP16_row_scale": "all +1", "FP32_column_scale": "all +1",
                    "FP16_initial_output": "all zero", "Real_target": 32}
    values = [value.data if isinstance(value, Tensor) else value for value in captured["args"]]
    return values, captured["kwargs"], {
        "kind": kind, "shape": {"M": m, "K": k, **({"N": n} if kind == "gemm" else {})},
        "contents": contents, "canonical_assertions_accepted": True,
        "canonical_launch_strides": strides, "configs": len(configs), "all_grids": grid_records,
        "events": events,
        "CPU_input_backing_bytes": [t.data.nbytes for t in inputs],
        "boundary": "Actual assertion/wrapper AST and CPU storage; mocked CUDA ownership/allocation/launch; no GPU execution.",
    }


def specialize(kernel, values, keywords, config, backend, unit_stride):
    from triton._C.libtriton import native_specialize_impl
    arguments = dict(zip(kernel.arg_names, values))
    arguments.update(keywords, **config.kwargs)
    bound = {name: arguments[name] for name in kernel.arg_names}
    specializations = [
        ("constexpr", arguments[p.name]) if p.is_constexpr else
        native_specialize_impl(backend, arguments[p.name], p.is_const,
                               not p.do_not_specialize, not p.do_not_specialize_on_alignment)
        for p in kernel.params
    ]
    options = {"num_warps": config.num_warps, "num_stages": config.num_stages,
               "num_ctas": config.num_ctas}
    _, signature, constants, attrs = kernel._pack_args(backend, options, bound, specializations, None)
    assert signature[unit_stride] == "constexpr"
    assert constants[(kernel.arg_names.index(unit_stride),)] == 1
    return signature, constants, attrs, options


def run(output):
    os.environ.pop("TRITON_INTERPRET", None)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(output / "tmp")
    os.environ["TRITON_CACHE_DIR"] = str(output / "compiler-cache")
    import triton
    from triton.compiler import ASTSource
    from triton.compiler.errors import CompilationError
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    assert pin(SOURCE)["sha256"] == SOURCE_SHA
    target = GPUTarget("cuda", 80, 32)
    backend = CUDABackend(target)
    cases, failures, controls = [], [], []
    for kind, indices, unit in [("producer", (3, 0), "stride_fpak"), ("gemm", (10, 27), "stride_ak")]:
        kernel, configs, tree = load_kernel(SOURCE, kind, "quantization_defect_" + kind)
        for m in (1, 2):
            values, keywords, admitted = admitted_call(tree, configs, kind, m, n=m)
            cases.append(admitted)
            expected = "M = M.to(tl.int64)" if m == 1 else f"{unit} = {unit}.to(tl.int64)"
            print(f"{kind}: canonical assertions admit {admitted['shape']}", flush=True)
            for index in indices:
                config = configs[index]
                signature, constants, attrs, options = specialize(kernel, values, keywords, config, backend, unit)
                try:
                    triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                                   target=target, options=options)
                except CompilationError as error:
                    message = str(error)
                    assert expected in message and "'int' object has no attribute 'to'" in message, message
                    path = output / f"{kind}_M{m}_config{index}.error.txt"
                    path.write_text(traceback.format_exc())
                    failures.append({
                        "kind": kind, "shape": admitted["shape"], "config_index": index,
                        "config": config.kwargs, "options": options, "signature": signature,
                        "constants": {kernel.arg_names[key[0]]: value for key, value in constants.items()},
                        "attrs": {kernel.arg_names[key[0]]: value for key, value in attrs.items()},
                        "error_type": type(error).__name__, "failed_source_expression": expected,
                        "error": message, "trace": pin(path),
                    })
                    print(f"{kind} M={m} config {index}: confirmed failure at {expected}", flush=True)
                else:
                    raise AssertionError("canonical source unexpectedly compiled")

        # Compile the unchanged original body with the same admitted M=2 call
        # and native specialization. Success is a compiler control only.
        original, original_configs, _ = load_kernel(ORIGINAL, kind, "quantization_control_" + kind)
        assert [c.kwargs for c in configs] == [c.kwargs for c in original_configs]
        index = indices[0]
        signature, constants, attrs, options = specialize(original, values, keywords, original_configs[index], backend, unit)
        compiled = triton.compile(ASTSource(original, signature=signature, constexprs=constants, attrs=attrs),
                                  target=target, options=options)
        artifacts = []
        for ext in ("ttir", "ttgir", "llir", "ptx"):
            path = output / f"original_{kind}_M2_config{index}.{ext}"
            path.write_text(compiled.asm[ext])
            artifacts.append(pin(path))
        controls.append({"kind": kind, "shape": admitted["shape"], "config_index": index,
                         "status": "offline_compile_success", "artifacts": artifacts})
        print(f"original {kind} M=2 config {index}: control compiled", flush=True)

    assert pin(SOURCE)["sha256"] == SOURCE_SHA
    record = {
        "task": "int8_matmul_quantization", "diagnostic_exit_code": 0,
        "classification": "canonical_checked_source_defect_pending_repair",
        "canonical_source": pin(SOURCE), "original_source": pin(ORIGINAL),
        "checker": pin(Path(__file__)), "triton_version": triton.__version__,
        "compiler_target": "cuda sm80 warp32", "admitted_cases": cases,
        "confirmed_compile_failures": failures, "original_controls": controls,
        "source_binding": "Exact canonical JIT body/signature and @triton.jit; only host autotune decorators omitted. Configurations and wrapper argument values extracted from canonical AST. Native ordinary specialization derives constants and signature.",
        "conclusion": "Both wrappers pass Python unit strides. Ordinary Triton specialization makes them constexpr integers, but the canonical JIT bodies invoke tensor .to on them. Admitted nonempty calls fail before producing outputs.",
        "numerical_assumptions": [], "deferred_experiments": [],
        "GPU_execution": False, "independent_numerical_replay": False,
        "preserved": "Canonical Python and input contract unchanged. Existing Lean model results remain valid within their stated interpretation; no full executable kernel correctness is claimed.",
        "prior_probe_limit": "The earlier explicit all-i64 signature probe and CPU checker using int64 tensor scalars bypassed ordinary Python-int specialization. Their success does not establish executable canonical correctness.",
    }
    path = output / "counterexample.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"confirmed_failures": len(failures), "control_compiles": len(controls),
                      "receipt": pin(path)}), flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".veritile-cache/task-deliveries/int8_matmul_quantization/constexpr-cast")
    run(parser.parse_args().output)
