"""Offline compilation of actual admitted launch arguments; no GPU execution."""
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / "kernel-evidence"
OUT.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(OUT / "cache")

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from triton.runtime.jit import create_function_from_signature

torch.set_num_threads(2)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = HERE / "rotary_transform_ops_repaired.py"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(SOURCE) == "c384a0bfe3d62579e9e4584a8e03d52bad3957c6790f55e30d493fb7f0350874"
actual = load("rotary_ops_selected_profiles", SOURCE)
fixtures = load("rotary_ops_profile_fixtures", HERE.parent / "rotary_transform/check_contract.py")
apply, events = fixtures.host(SOURCE)


class Capture:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.args = [a.data if isinstance(a, fixtures.Tensor) else a for a in args]
            self.kwargs = kwargs
            self.grid = grid(dict(zip(actual.rotary_kernel.arg_names, args)))
            events.append(("captured_launch", self.grid))
        return launch


capture = Capture()
apply.__globals__["rotary_kernel"] = capture
profiles = list(itertools.product((False, True), repeat=4))
records = []
for varlen, offsets_tensor, interleaved, conjugate in profiles:
    width = 130 if interleaved else 66
    cu_type = torch.int64 if conjugate else torch.int32
    offsets_type = torch.int32 if conjugate else torch.int64
    shape = (4, 2, width) if varlen else (3, 3, 2, width)
    x = fixtures.Tensor(torch.ones(shape))
    co = fixtures.Tensor(torch.full((4, width // 2), 2.0))
    sn = fixtures.Tensor(torch.ones(4, width // 2))
    cu = fixtures.Tensor(torch.tensor([0, 1, 1, 4], dtype=cu_type)) if varlen else None
    offsets = fixtures.Tensor(torch.tensor([0, 7, 1], dtype=offsets_type)) if offsets_tensor else 0
    events.clear()
    apply(x, co, sn, offsets, cu, 3 if varlen else None, interleaved, False, conjugate)
    name = f"v{int(varlen)}_t{int(offsets_tensor)}_i{int(interleaved)}_c{int(conjugate)}"
    target = GPUTarget("cuda", 80, 32)
    backend = CUDABackend(target)
    fn = actual.rotary_kernel
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, options = binder(*capture.args, **capture.kwargs)
    parsed, signature, constants, attrs = fn._pack_args(backend, options, bound, specialization, options)
    row = dict(name=name, source_sha256=sha(SOURCE), grid=capture.grid, signature=signature,
               constants={fn.arg_names[k[0]]: str(v) for k, v in constants.items()},
               options=repr(parsed), events=list(events))
    try:
        compiled = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
                                  target=target, options=options)
    except Exception as error:
        path = OUT / (name + ".error.txt")
        path.write_text(traceback.format_exc())
        row.update(status="failed", error=str(error), artifact=str(path))
    else:
        artifacts = []
        for kind in ("ttir", "ptx"):
            path = OUT / (name + "." + kind)
            path.write_text(compiled.asm[kind])
            artifacts.append(dict(path=str(path), sha256=sha(path)))
        row.update(status="compiled", artifacts=artifacts)
    records.append(row)
    print(name, row["status"], row.get("error", ""), flush=True)
    (OUT / "profiles.json").write_text(json.dumps(dict(triton=triton.__version__,
        gpu_execution=False, actual_binder=True, actual_wrapper_arguments=True,
        results=records), indent=2) + "\n")

assert all(row["status"] == "compiled" for row in records)
print("PASS: 16 complete source profiles, both original integer metadata widths, default SM80 compilation.")
