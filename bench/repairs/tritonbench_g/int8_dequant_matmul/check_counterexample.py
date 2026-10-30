"""Admitted-input compiler counterexample for constexpr scalar `.to` calls.

Runs the actual canonical assertions/wrapper with CPU-backed tensors and mocked
CUDA launch, derives the ordinary native JIT argument specialization, and
checks the resulting exact-source offline compilation errors. No GPU executes.
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
SOURCE = HERE / "int8_dequant_matmul_repaired.py"
ORIGINAL = ROOT / "bench/tritonbench_g/int8_dequant_matmul/int8_dequant_matmul.py"
SOURCE_SHA = "d71b15843752bfa2904d3d341a49ce9017635a7a4273acf1ed4d7edb4620c44f"
KERNEL = "_int8_matmul_rowwise_dequantize"


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def function(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def load_kernel(path, module_name):
    import torch
    import triton
    import triton.language as tl
    tree = ast.parse(path.read_text())
    node = copy.deepcopy(function(tree, KERNEL))
    configs_expr = next(k.value for k in node.decorator_list[0].keywords if k.arg == "configs")
    # Only the two host decorators are omitted. The unchanged @triton.jit,
    # function body, parameter annotations, and original filename remain.
    # This avoids importing the unavailable legacy host performance model;
    # no autotuning or pruning is exercised by this kernel-only diagnostic.
    node.decorator_list = [node.decorator_list[-1]]
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, triton=triton, tl=tl)
    sys.modules[module_name] = module
    nodes = [function(tree, "get_configs_io_bound"), node]
    exec(compile(ast.Module(nodes, []), str(path), "exec"), module.__dict__)
    configs = eval(compile(ast.Expression(configs_expr), str(path), "eval"), module.__dict__)
    kernel = getattr(module, KERNEL)
    actual = ast.parse(kernel.src).body[0]
    assert [ast.dump(n) for n in actual.body] == [ast.dump(n) for n in node.body]
    assert [ast.dump(n) for n in actual.args.args] == [ast.dump(n) for n in node.args.args]
    return kernel, configs, tree


def admitted_call(tree, configs, m, n, k):
    import torch
    import triton
    import triton.language as tl

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

    def empty(shape, device, dtype):
        events.append("allocation")
        assert shape == (m, n) and dtype == torch.float16
        return Tensor(torch.empty(shape, dtype=dtype))

    class Launch:
        def __getitem__(self, grid):
            captured["grid"] = grid

            def call(*args, **kwargs):
                events.append("launch")
                captured.update(args=args, kwargs=kwargs)

            return call

    fake = types.SimpleNamespace(
        Tensor=Tensor, strided=torch.strided, int8=torch.int8,
        float32=torch.float32, float16=torch.float16, empty=empty,
        cuda=types.SimpleNamespace(device=device),
        _C=types.SimpleNamespace(_overlaps=lambda a, b: torch._C._overlaps(a.data, b.data)),
    )
    namespace = {"torch": fake, "triton": triton, "tl": tl, KERNEL: Launch()}
    nodes = [function(tree, "_check"), function(tree, "int8_matmul_rowwise_dequantize")]
    exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), namespace)
    inputs = [Tensor(torch.ones(shape, dtype=dtype)) for shape, dtype in
              [((m, k), torch.int8), ((k, n), torch.int8),
               ((m,), torch.float32), ((n,), torch.float32)]]
    result = namespace["int8_matmul_rowwise_dequantize"](*inputs, None)
    assert events == ["allocation", "device-enter", "launch", "device-exit"]
    assert result is captured["args"][2]
    values = [value.data if isinstance(value, Tensor) else value for value in captured["args"]]
    assert values[6:9] == [m, n, k]
    assert values[11:17] == [k, 1, n, 1, n, 1]
    assert len(configs) == 418
    grids = [list(captured["grid"](cfg.kwargs)) for cfg in configs]
    assert all(grid == [1, cfg.kwargs["SPLIT_K"]] for grid, cfg in zip(grids, configs))
    return values, captured["kwargs"], {
        "shape": {"M": m, "N": n, "K": k},
        "A_INT8": "all +1", "B_INT8": "all +1",
        "row_scale_FP32": "all +1", "column_scale_FP32": "all +1", "bias": None,
        "canonical_assertions_accepted": True,
        "canonical_launch_strides": values[11:17],
        "configurations": len(configs), "all_grids": grids,
        "events": events,
        "CPU_backing_bytes": [tensor.data.nbytes for tensor in inputs] + [result.data.nbytes],
        "boundary": "Actual canonical assertion and wrapper AST; full real CPU contents; mocked CUDA ownership/allocation/launch; no device kernel execution.",
    }


def specialize(kernel, values, keywords, config, backend):
    from triton._C.libtriton import native_specialize_impl
    arguments = dict(zip(kernel.arg_names[:17], values))
    arguments.update(keywords, **config.kwargs)
    arguments["EVEN_K"] = arguments["K"] % (config.kwargs["BLOCK_K"] * config.kwargs["SPLIT_K"]) == 0
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
    assert signature["stride_ak"] == signature["stride_bn"] == signature["stride_cn"] == "constexpr"
    assert constants[(kernel.arg_names.index("stride_ak"),)] == 1
    return signature, constants, attrs, options


def run(output):
    os.environ.pop("TRITON_INTERPRET", None)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "compiler-cache")
    import triton
    from triton.compiler import ASTSource
    from triton.compiler.errors import CompilationError
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    assert pin(SOURCE)["sha256"] == SOURCE_SHA
    kernel, configs, tree = load_kernel(SOURCE, "dequant_counterexample_canonical")
    target = GPUTarget("cuda", 90, 32)
    backend = CUDABackend(target)
    cases, failures = [], []
    for m, n, expected in [(1, 1, "M = M.to(tl.int64)"), (2, 2, "stride_ak = stride_ak.to(tl.int64)")]:
        values, keywords, admitted = admitted_call(tree, configs, m, n, 32)
        cases.append(admitted)
        print(f"canonical assertions admit M={m}, N={n}, K=32; all 418 grids valid", flush=True)
        for index in (18, 6):
            config = configs[index]
            signature, constants, attrs, options = specialize(kernel, values, keywords, config, backend)
            try:
                triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                               target=target, options=options)
            except CompilationError as error:
                message = str(error)
                assert expected in message and "'int' object has no attribute 'to'" in message, message
                path = output / f"M{m}_N{n}_config{index}.error.txt"
                path.write_text(traceback.format_exc())
                failures.append({
                    "shape": admitted["shape"], "config_index": index,
                    "config": config.kwargs, "options": options,
                    "signature": signature,
                    "constants": {kernel.arg_names[key[0]]: str(value) if kernel.arg_names[key[0]] == "ACC_TYPE" else value
                                  for key, value in constants.items()},
                    "attrs": {kernel.arg_names[key[0]]: value for key, value in attrs.items()},
                    "error_type": type(error).__name__, "failed_source_expression": expected,
                    "error": message, "trace": pin(path),
                })
                print(f"config {index}: confirmed CompilationError at {expected}", flush=True)
            else:
                raise AssertionError("canonical source unexpectedly compiled")

    # One targeted control: original source body, same admitted launch values
    # and ordinary specialization. This is compiler success, not correctness.
    original, original_configs, _ = load_kernel(ORIGINAL, "dequant_counterexample_original")
    assert [c.kwargs for c in configs] == [c.kwargs for c in original_configs]
    signature, constants, attrs, options = specialize(original, values, keywords, original_configs[18], backend)
    compiled = triton.compile(ASTSource(original, signature=signature, constexprs=constants, attrs=attrs),
                              target=target, options=options)
    artifacts = []
    for ext in ("ttir", "ttgir", "llir", "ptx"):
        path = output / f"original_M2_N2_config18.{ext}"
        path.write_text(compiled.asm[ext])
        artifacts.append(pin(path))
    print("original source control M=N=2, K=32, config18 compiled successfully", flush=True)
    record = {
        "task": "int8_dequant_matmul", "diagnostic_exit_code": 0,
        "classification": "canonical_checked_source_defect_pending_repair",
        "canonical_source": pin(SOURCE), "original_source": pin(ORIGINAL),
        "checker": pin(Path(__file__)), "triton_version": triton.__version__,
        "compiler_target": "cuda sm90 warp32",
        "source_binding": "Unchanged canonical kernel body, signature and @triton.jit. Host autotune/heuristics decorators omitted for offline compilation; config objects and wrapper argument values extracted from their exact canonical AST. Legacy host performance-model import is not executed.",
        "admitted_cases": cases, "confirmed_compile_failures": failures,
        "original_control": {"shape": {"M": 2, "N": 2, "K": 32}, "config_index": 18,
                             "status": "offline_compile_success", "artifacts": artifacts,
                             "scope": "No correctness, hardware execution, or original-benchmark acceptance claim."},
        "conclusion": "The canonical wrapper always supplies unit strides. Ordinary Triton JIT specialization makes them constexpr integers, but the canonical kernel invokes tensor .to on them. An admitted configured call therefore fails before producing an output. The source's successful-execution target does not hold for this canonical revision.",
        "numerical_assumptions": [], "deferred_experiments": [],
        "GPU_execution": False, "independent_numerical_replay": False,
        "preserved": "No canonical source, assertion, configuration or existing Lean theorem changed. Prior mathematical and abstract source-plan results retain their declared interpretation. The earlier carry-saturation hypothesis is not asserted as a successfully compiled canonical execution.",
    }
    path = output / "counterexample.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"confirmed_failures": len(failures), "control_compiles": 1,
                      "receipt": pin(path)}), flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".veritile-cache/kernel-evidence/int8_dequant_matmul/constexpr-cast")
    run(parser.parse_args().output)
