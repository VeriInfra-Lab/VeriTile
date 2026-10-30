"""Source-specific small counterexamples; CPU AST replay, never a GPU claim.

Real PyTorch DLPack aliases retain their real pointers/storage identities.
Only CUDA admission/device contexts and Triton operations use CPU adapters.
Large tile cases stop in the actual Triton frontend, before optimization.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import resource
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "bgmv_shrink_kernel_repaired.py"
PIN = "7152c240f53ba0c35f44fc3967491bb31e02c60a0cbe235cafdcfacaa0c3018d"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(name):
    return copy.deepcopy(next(n for n in ast.parse(SOURCE.read_text()).body
                              if isinstance(n, ast.FunctionDef) and n.name == name))


def run(out):
    # Set all temporary/compiler paths before importing the compiler or Torch.
    out.mkdir(parents=True, exist_ok=True)
    for name, suffix in (("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton-cache"),
                         ("TRITON_DUMP_DIR", "triton-dump"),
                         ("TRITON_OVERRIDE_DIR", "triton-override")):
        directory = out / suffix
        directory.mkdir(exist_ok=True)
        os.environ[name] = str(directory)
    import torch
    import triton
    torch.set_num_threads(1)
    assert sha(SOURCE) == PIN

    class View:
        def __init__(self, data):
            self.data = data
            self.device = torch.device("cuda", 0)
            self.is_cuda = True

        def __getattr__(self, key):
            return getattr(self.data, key)

        def __eq__(self, other):
            return self.data == other

        def __ge__(self, other):
            return self.data >= other

        def __lt__(self, other):
            return self.data < other

        def squeeze(self, dim):
            return View(self.data.squeeze(dim))

    class Pointer:
        def __init__(self, tensor, label, offset=0):
            self.tensor, self.label, self.offset = tensor, label, offset

        def __add__(self, offset):
            return Pointer(self.tensor, self.label, self.offset + offset)

    def raw(tensor):
        storage = tensor.untyped_storage()
        return torch.empty(0, dtype=tensor.dtype).set_(
            storage, 0, (storage.nbytes() // tensor.element_size(),), (1,))

    def byte_backing(tensor):
        storage = tensor.untyped_storage()
        return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))

    class ActiveAccessError(Exception):
        pass

    pid = [0, 0]
    events = []

    def access(pointer, mask):
        offsets = torch.as_tensor(pointer.offset, dtype=torch.int64)
        mask = (torch.ones_like(offsets, dtype=torch.bool) if mask is None else
                torch.broadcast_to(torch.as_tensor(mask, dtype=torch.bool), offsets.shape))
        indices = pointer.tensor.storage_offset() + offsets[mask]
        capacity = raw(pointer.tensor).numel()
        if indices.numel() and (indices.min() < 0 or indices.max() >= capacity):
            event = {"kind": "active_access_outside_backing", "operand": pointer.label,
                     "program": pid.copy(), "first_index": indices.min().item(),
                     "last_index": indices.max().item(), "capacity": capacity}
            events.append(event)
            raise ActiveAccessError(event)
        return offsets, mask, indices

    def load(pointer, mask=None, other=0):
        offsets, mask, indices = access(pointer, mask)
        result = torch.full_like(offsets, other, dtype=pointer.tensor.dtype)
        result[mask] = raw(pointer.tensor)[indices]
        if pointer.label == "indices":
            events.append({"kind": "metadata_load", "program": pid.copy(), "value": result.item()})
        return result

    def write(pointer, value, mask=None, atomic=False):
        offsets, mask, indices = access(pointer, mask)
        values = torch.broadcast_to(value, offsets.shape)[mask].to(pointer.tensor.dtype)
        backing = raw(pointer.tensor)
        if atomic:
            backing[indices] += values
        else:
            backing[indices] = values
        events.append({"kind": "atomic" if atomic else "store", "program": pid.copy(),
                       "indices": indices.tolist(), "payload": values.tolist()})

    tl = SimpleNamespace(
        program_id=lambda axis: torch.tensor(pid[axis]), arange=torch.arange,
        int64=torch.int64, float32=torch.float32, cast=lambda x, dtype: torch.as_tensor(x, dtype=dtype),
        zeros=torch.zeros, max_contiguous=lambda x, n: x,
        load=load, store=lambda p, v, mask=None: write(p, v, mask),
        atomic_add=lambda p, v, mask=None: write(p, v, mask, True),
        sum=lambda x, axis: x.sum(dim=axis, dtype=x.dtype))
    kernel_node = function("_bgmv_shrink_kernel")
    kernel_node.decorator_list = []
    for arg in kernel_node.args.args:
        arg.annotation = None
    kernel_scope = {"tl": tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[kernel_node], type_ignores=[])),
                 str(SOURCE), "exec"), kernel_scope)
    kernel = kernel_scope["_bgmv_shrink_kernel"]

    launches = []
    capture_only = [False]

    class Launch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append({"grid": list(grid), "N": args[3], "K": args[4],
                                 "scaling": args[6], "strides": list(args[7:]), **kwargs})
                if capture_only[0]:
                    return
                labels = {0: "X", 1: "W", 2: "Y", 5: "indices"}
                values = [Pointer(arg.data, labels[j]) if isinstance(arg, View) else arg
                          for j, arg in enumerate(args)]
                for batch in range(grid[1]):
                    for split in range(grid[0]):
                        pid[:] = [split, batch]
                        kernel(*values, **kwargs)
            return launch

    host_torch = SimpleNamespace(Tensor=View, strided=torch.strided,
                                float16=torch.float16, bfloat16=torch.bfloat16,
                                float32=torch.float32, int32=torch.int32, int64=torch.int64,
                                all=torch.all, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    host_scope = {"torch": host_torch, "triton": triton, "math": math,
                  "_bgmv_shrink_kernel": Launch()}
    host_node = function("_bgmv_shrink")
    host_node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[host_node], type_ignores=[])),
                 str(SOURCE), "exec"), host_scope)
    invoke = host_scope["_bgmv_shrink"]

    def actual_call(tensors, scale=1.0, capture=False):
        launches.clear()
        events.clear()
        capture_only[0] = capture
        return invoke(*(View(t) for t in tensors), scale)

    def snapshot_target(tensors, scale=1.0):
        # Independent caller-coordinate polynomial, computed before execution.
        x, w, y, indices = [t.clone() for t in tensors]
        w = w.squeeze(1) if w.ndim == 4 else w
        result = y.double()
        for batch, adapter in enumerate(indices.tolist()):
            if adapter != -1:
                for col in range(w.shape[1]):
                    result[batch, col] += scale * sum(
                        x[batch, k].item() * w[adapter, col, k].item() for k in range(w.shape[2]))
        return result

    alias_cases = []
    for dtype, word in ((torch.float16, 0x3F803C00), (torch.bfloat16, 0x3F803F80)):
        for operand in ("X", "W"):
            for index_type in (torch.int32, torch.int64):
                for four_dim in (False, True):
                    shared = torch.zeros(264, dtype=dtype)
                    y = torch.utils.dlpack.from_dlpack(shared).view(torch.float32)[128:129].reshape(1, 1)
                    y.view(torch.int32).fill_(word)
                    shared[0] = 1
                    source_view = shared[:257]
                    x = (source_view if operand == "X" else torch.ones(257, dtype=dtype)).reshape(1, 257)
                    w = (source_view if operand == "W" else torch.ones(257, dtype=dtype)).reshape(
                        (1, 1, 1, 257) if four_dim else (1, 1, 257))
                    indices = torch.zeros(1, dtype=index_type)
                    tensors = [x, w, y, indices]
                    read = x if operand == "X" else w
                    assert read.untyped_storage()._cdata != y.untyped_storage()._cdata
                    assert y.data_ptr() == read.data_ptr() + 512
                    assert read.data_ptr() % 16 == y.data_ptr() % 16 == 0
                    target = snapshot_target(tensors).item()
                    before = byte_backing(read).clone()
                    initial = y.item()
                    control = [x.clone(), w.clone(), y.clone(), indices.clone()]
                    actual_call(control)
                    assert control[2].item() == target
                    assert len([e for e in events if e["kind"] == "atomic"]) == 64
                    actual_call(tensors)
                    assert len(launches) == 1 and launches[0]["grid"] == [64, 1]
                    assert len([e for e in events if e["kind"] == "atomic"]) == 64
                    assert y.item() != target
                    assert not torch.equal(byte_backing(read), before)
                    alias_cases.append({"input_dtype": str(dtype), "read_operand": operand,
                                        "metadata_dtype": str(index_type), "weight_rank": w.ndim,
                                        "different_storage_ids": True, "same_backing_bytes": True,
                                        "output_pointer_delta": 512, "initial_output": initial,
                                        "snapshot_target": target, "actual_source_output": y.item(),
                                        "disjoint_control_output": control[2].item(),
                                        "atomic_count": 64, "read_backing_frame_violated": True,
                                        "launch": launches[0], "first_four_events": events[:4]})

    metadata_cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for index_type in (torch.int32, torch.int64):
            x = torch.zeros(1, 257, dtype=dtype)
            x[0, 0] = x[0, 256] = 1
            w = torch.ones(1, 1, 257, dtype=dtype)
            indices = torch.zeros(1, dtype=index_type)
            y = torch.utils.dlpack.from_dlpack(indices).view(torch.float32)[:1].reshape(1, 1)
            assert indices.untyped_storage()._cdata != y.untyped_storage()._cdata
            assert indices.data_ptr() == y.data_ptr()
            tensors = [x, w, y, indices]
            control = [x.clone(), w.clone(), y.clone(), indices.clone()]
            target = snapshot_target(tensors).item()
            actual_call(control)
            assert control[2].item() == target == 2.0 and control[3].item() == 0
            try:
                actual_call(tensors)
            except ActiveAccessError:
                pass
            else:
                raise AssertionError("metadata alias must reach an active out-of-bounds weight read")
            assert len(launches) == 1
            assert indices.item() == 1065353216 and y.item() == 1.0
            assert events[-1]["operand"] == "W" and events[-1]["program"] == [1, 0]
            metadata_cases.append({"input_dtype": str(dtype), "metadata_dtype": str(index_type),
                                   "initial_metadata": 0, "metadata_after_first_atomic": indices.item(),
                                   "snapshot_target": target, "disjoint_control_output": control[2].item(),
                                   "different_storage_ids": True, "equal_data_pointers": True,
                                   "events": events.copy(), "launch": launches[0]})

    controls = []
    for operand in ("X", "W", "indices"):
        data = torch.zeros(264, dtype=torch.float16)
        x, w = data[:257].reshape(1, 257), torch.ones(1, 1, 257, dtype=torch.float16)
        indices = torch.zeros(1, dtype=torch.int32)
        if operand == "X":
            y = data.view(torch.float32)[128:129].reshape(1, 1)
        elif operand == "W":
            w = data[:257].reshape(1, 1, 257)
            x = torch.ones(1, 257, dtype=torch.float16)
            y = data.view(torch.float32)[128:129].reshape(1, 1)
        else:
            y = indices.view(torch.float32).reshape(1, 1)
        try:
            actual_call([x, w, y, indices])
        except ValueError as error:
            assert "output backing" in str(error) and not launches
        else:
            raise AssertionError("same StorageImpl alias must be rejected")
        controls.append({"case": "same_cdata_" + operand, "rejected_before_launch": True})
    for dtype in (torch.float16, torch.bfloat16):
        for k, slot, scale in ((0, 0, 2.0), (257, -1, 1.0), (257, 0, 0.5)):
            x = torch.zeros(1, k, dtype=dtype)
            if k:
                x[0, 0] = x[0, 256] = 1
            tensors = [x, torch.ones(1, 1, k, dtype=dtype), torch.full((1, 1), 7.0),
                       torch.tensor([slot], dtype=torch.int64)]
            target = snapshot_target(tensors, scale).item()
            actual_call(tensors, scale)
            assert tensors[2].item() == target
            count = len([e for e in events if e["kind"] == "atomic"])
            assert count == (0 if slot == -1 else 64)
            controls.append({"case": "disjoint_edge", "dtype": str(dtype), "K": k,
                             "adapter": slot, "scale": scale, "initial_output": 7.0,
                             "output": target, "atomic_count": count})
    for b, n in ((0, 1), (1, 0)):
        tensors = [torch.zeros(b, 1, dtype=torch.float16), torch.ones(1, n, 1, dtype=torch.float16),
                   torch.zeros(b, n), torch.full((b,), -2, dtype=torch.int32)]
        actual_call(tensors)
        assert not launches
        controls.append({"case": "empty_before_metadata_validation", "B": b, "N": n})

    # Capture genuine admitted launch arguments with small actual allocations.
    # The installed frontend rejects the oversized type before producing it.
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend
    from triton.runtime.jit import create_function_from_signature
    from triton._C.libtriton import ir
    from triton._utils import validate_block_shape
    module_spec = importlib.util.spec_from_file_location("bgmv_shrink_boundary_source", SOURCE)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    jit = module._bgmv_shrink_kernel
    target = GPUTarget("cuda", 80, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4})
    signature = {name: "i32" for name in jit.arg_names[:14]}
    signature.update(input_ptr="*fp16", lora_ptr="*fp16", out_ptr="*fp32",
                     lora_indices="*i32", scaling="fp32")
    compiler_cases = []
    for n in (3, 4096, 4097):
        tensors = [torch.ones(1, 1, dtype=torch.float16), torch.ones(1, n, 1, dtype=torch.float16),
                   torch.zeros(1, n), torch.zeros(1, dtype=torch.int32)]
        actual_call(tensors, capture=True)
        captured = launches[0].copy()
        constants = {name: captured[name] for name in ("BLOCK_N", "BLOCK_K", "SPLIT_K")}
        src = ASTSource(jit, signature, constexprs=constants)
        ctx = ir.context()
        ir.load_dialects(ctx)
        backend.load_dialects(ctx)
        record = {"N": n, "launch": captured, "tensor_numel": captured["BLOCK_N"] * 256,
                  "phase": "actual ASTSource.make_ir frontend only", "optimization_run": False}
        try:
            result = src.make_ir(target, options, backend.get_codegen_implementation(options),
                                 backend.get_module_map(), ctx)
        except Exception as error:
            assert n == 4097 and "numel (2097152) exceeds triton maximum tensor numel (1048576)" in str(error)
            record.update(result="rejected", exception=type(error).__name__, message=str(error))
            (out / "tile-4097-error.txt").write_text(str(error) + "\n")
        else:
            assert n in (3, 4096)
            text = str(result)
            path = out / f"frontend-n{n}.ttir"
            path.write_text(text)
            record.update(result="accepted", ttir={"path": str(path), "sha256": sha(path)})
        compiler_cases.append(record)
    # Invoke the same generated binder and _pack_args used by JITFunction.run,
    # with the selected CUDA backend. CPU tensors supply genuine dtype/pointer
    # metadata; this does not create a GPU device, stream, or launch.
    binder = create_function_from_signature(jit.signature, jit.params, backend)
    binder_cases = []
    for n in (3, 4096, 4097):
        tensors = [torch.ones(1, 1, dtype=torch.float16), torch.ones(1, n, 1, dtype=torch.float16),
                   torch.zeros(1, n), torch.zeros(1, dtype=torch.int32)]
        actual_call(tensors, capture=True)
        captured = launches[0].copy()
        constants = {name: captured[name] for name in ("BLOCK_N", "BLOCK_K", "SPLIT_K")}
        args = [tensors[0], tensors[1], tensors[2], n, 1, tensors[3], captured["scaling"],
                *captured["strides"]]
        bound, specialization, extra = binder(*args, **constants)
        packed_options, packed_sig, packed_constants, packed_attrs = jit._pack_args(
            backend, constants, bound, specialization, extra)
        assert packed_sig["scaling"] == "fp32" and packed_sig["K"] == "constexpr"
        src = ASTSource(jit, packed_sig, constexprs=packed_constants, attrs=packed_attrs)
        ctx = ir.context()
        ir.load_dialects(ctx)
        backend.load_dialects(ctx)
        record = {"N": n, "launch": captured, "signature": packed_sig,
                  "specialization": dict(zip(jit.arg_names, specialization)),
                  "constexprs": {jit.arg_names[key[0]]: val for key, val in packed_constants.items()},
                  "attrs": {jit.arg_names[key[0]]: val for key, val in packed_attrs.items()},
                  "phase": "actual generated binder/_pack_args then ASTSource.make_ir only",
                  "optimization_run": False, "gpu_runtime_used": False}
        try:
            result = src.make_ir(target, packed_options, backend.get_codegen_implementation(packed_options),
                                 backend.get_module_map(), ctx)
        except Exception as error:
            assert n == 4097 and "numel (2097152) exceeds triton maximum tensor numel (1048576)" in str(error)
            record.update(result="rejected", exception=type(error).__name__, message=str(error))
            (out / "binder-tile-4097-error.txt").write_text(str(error) + "\n")
        else:
            assert n in (3, 4096)
            path = out / f"binder-frontend-n{n}.ttir"
            path.write_text(str(result))
            record.update(result="accepted", ttir={"path": str(path), "sha256": sha(path)})
        binder_cases.append(record)
    scalar_abi = []
    for dtype in (torch.float16, torch.bfloat16):
        for index_type in (torch.int32, torch.int64):
            for scale in (1.0, 1e-50, 1e300):
                args = [torch.ones(1, 257, dtype=dtype), torch.ones(1, 1, 257, dtype=dtype),
                        torch.zeros(1, 1), 1, 257, torch.zeros(1, dtype=index_type), scale,
                        257, 1, 257, 257, 1, 1, 1]
                _, spec, _ = binder(*args, BLOCK_N=1, BLOCK_K=256, SPLIT_K=64)
                assert spec[6][0] == "fp32"
                scalar_abi.append({"input_dtype": str(dtype), "metadata_dtype": str(index_type),
                                   "finite_python_scaling": scale,
                                   "signature": dict(zip(jit.arg_names, [s[0] for s in spec]))})
    (out / "triton-binder-sources.py").write_text(
        inspect.getsource(create_function_from_signature) + "\n" + inspect.getsource(type(jit)._pack_args))
    # Full lowering controls use small tiles only, for both input/index formats.
    ptx_cases = []
    for fmt in ("fp16", "bf16"):
        for index_fmt in ("i32", "i64"):
            sig = dict(signature, input_ptr="*" + fmt, lora_ptr="*" + fmt, lora_indices="*" + index_fmt)
            src = ASTSource(jit, sig, constexprs={"BLOCK_N": 4, "BLOCK_K": 256, "SPLIT_K": 64})
            compiled = triton.compile(src, target=target, options={"num_warps": 4})
            ptx = compiled.asm["ptx"]
            ttir = compiled.asm["ttir"]
            half_type = "f16" if fmt == "fp16" else "bf16"
            assert any("arith.mulf" in line and half_type in line for line in ttir.splitlines())
            assert any("arith.addf" in line and half_type in line for line in ttir.splitlines())
            assert "tt.atomic_rmw fadd" in ttir and "!tt.ptr<f32>" in ttir
            atomic_instructions = re.findall(r"\b(?:atom|red)\.[^\s]*\.add\.f32\b", ptx)
            assert atomic_instructions
            path = out / f"small-{fmt}-{index_fmt}.ptx"
            path.write_text(ptx)
            ir_path = out / f"small-{fmt}-{index_fmt}.ttir"
            ir_path.write_text(ttir)
            ptx_cases.append({"input_dtype": fmt, "metadata_dtype": index_fmt,
                              "BLOCK_N": 4, "BLOCK_K": 256, "scaling_signature": "fp32",
                              "atomic_instructions": sorted(set(atomic_instructions)),
                              "half_product_and_reduction_joins_checked": True,
                              "ttir": {"path": str(ir_path), "sha256": sha(ir_path)},
                              "ptx": {"path": str(path), "sha256": sha(path)}})
    checker_source = inspect.getsource(validate_block_shape)
    (out / "triton-validate-block-shape.py").write_text(checker_source)
    report = {"source": str(SOURCE), "source_sha256": PIN, "torch": torch.__version__,
              "triton": triton.__version__, "gpu_execution": False,
              "independent_target": "Y_before[b,n] + scale * sum_k X_before[b,k] * W_before[index_before[b],n,k]",
              "alias_cases": alias_cases, "metadata_alias_cases": metadata_cases,
              "positive_controls": controls, "compiler_cases": compiler_cases,
              "binder_frontend_cases": binder_cases, "scalar_abi_observations": scalar_abi,
              "small_full_ptx_controls": ptx_cases,
              "frontend_shape_check": {"path": str(out / "triton-validate-block-shape.py"),
                                       "sha256": sha(out / "triton-validate-block-shape.py")},
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "limitations": ["Real CPU Torch DLPack aliases; CUDA device admission/context is adapted.",
                              "Actual extracted source runs sequential split order 0..63; no GPU execution or arbitrary native numerical equivalence claim.",
                              "Counterexample reductions each have one nonzero exact product; disjoint replay agrees with the independent target.",
                              "Offline ASTSource signatures are explicit and do not prove actual runtime binder signature selection.",
                              "Separate binder-derived cases use the actual generated binder/_pack_args with an explicit CUDA backend and real CPU Tensor metadata; device/stream/launch remain unexecuted.",
                              "Large tile checks stop at frontend type construction; no large-tile optimization or PTX compilation is attempted."]}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out / "report.json"), "source_sha256": PIN,
                      "alias_output_counterexamples": len(alias_cases), "metadata_oob_counterexamples": len(metadata_cases),
                      "positive_controls": len(controls), "compiler_cases": len(compiler_cases),
                      "binder_frontend_cases": len(binder_cases), "scalar_abi_observations": len(scalar_abi),
                      "small_ptx_controls": len(ptx_cases), "peak_rss_kib": report["peak_rss_kib"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".veritile-cache/bgmv-shrink-kernel-evidence/replay")
    run(parser.parse_args().output_dir.resolve())
