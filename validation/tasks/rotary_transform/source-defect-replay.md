# Reproduce the admitted block-size failure

Run the following block with Python from the repository root. It reads the
unchanged canonical source, uses the existing CPU Tensor facade only to reach
the wrapper's launch with checked descriptors, and sends the captured arguments
through Triton's actual JIT binder and CUDA compiler. It does not execute GPU
code. The two small-width controls compile with the same default options.
No source file or admission check is edited.

```python
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
root = Path("bench/tritonbench_g/rotary_transform").resolve()
out = root / "source-defect-evidence"
out.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(out / "cache")

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from triton.runtime.jit import create_function_from_signature
from triton._utils import TRITON_MAX_TENSOR_NUMEL

torch.set_num_threads(2)
source = root / "rotary_transform_repaired.py"
original = root.parents[2] / "tritonbench_g/rotary_transform/rotary_transform.py"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(source) == "d59d9684bd9131a12e800b0a54859f06001de3f5b60f78b1e9f9eb625259eda8"
assert sha(original) == "920032ade0eb2da9a2cb26bbda36318ab1f74260be6b5b58f22427726a6399de"
assert TRITON_MAX_TENSOR_NUMEL == 1048576

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

actual = load("rotary_block_size_source", source)
fixtures = load("rotary_block_size_fixtures", root / "check_contract.py")
assert ast.dump(fixtures.f.function(source, "rotary_kernel")) == ast.dump(
    fixtures.f.function(original, "rotary_kernel"))
apply, events = fixtures.host(source)

class Capture:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.args = [a.data if isinstance(a, fixtures.Tensor) else a for a in args]
            self.kwargs = kwargs
            self.grid = grid(dict(zip(actual.rotary_kernel.arg_names, args)))
            events.append(("launch_captured", self.grid))
        return launch

capture = Capture()
apply.__globals__["rotary_kernel"] = capture

def observe(value):
    if not isinstance(value, torch.Tensor):
        return value
    storage = value.untyped_storage()
    return dict(shape=list(value.shape), strides=list(value.stride()),
                dtype=str(value.dtype), storage_offset=value.storage_offset(),
                data_ptr=value.data_ptr(), allocation_base=storage.data_ptr(),
                allocation_bytes=storage.nbytes(), contiguous=value.is_contiguous())

records = []
for interleaved, width, failure_expected in [
    (True, 262146, True), (False, 524290, True),
    (True, 514, False), (False, 514, False),
]:
    x = fixtures.Tensor(torch.ones(1, 1, 1, width))
    cosine = fixtures.Tensor(torch.ones(1, width // 2))
    sine = fixtures.Tensor(torch.zeros(1, width // 2))
    events.clear()
    apply(x, cosine, sine, interleaved=interleaved)
    assert capture.grid == (1, 1, 1)
    params = dict(zip(actual.rotary_kernel.arg_names, capture.args))
    assert params["BLOCK_M"] == 4
    assert params["BLOCK_K"] == triton.next_power_of_2(width)
    elements = 4 * (params["BLOCK_K"] if interleaved else params["BLOCK_K"] // 2)
    assert (elements > TRITON_MAX_TENSOR_NUMEL) == failure_expected
    for sm in (80, 90):
        name = f"sm{sm}_{'interleaved' if interleaved else 'split'}_width{width}"
        target = GPUTarget("cuda", sm, 32)
        backend = CUDABackend(target)
        fn = actual.rotary_kernel
        binder = create_function_from_signature(fn.signature, fn.params, backend)
        bound, specialization, options = binder(*capture.args, **capture.kwargs)
        parsed, signature, constants, attrs = fn._pack_args(
            backend, options, bound, specialization, options)
        row = dict(name=name, sm=sm, interleaved=interleaved, width=width,
                   failure_expected=failure_expected, grid=list(capture.grid),
                   pointer_elements=elements, events=list(events),
                   admitted_arguments={k: observe(v) for k, v in params.items()},
                   signature=signature, options=options,
                   parsed_options=repr(parsed),
                   constants={fn.arg_names[k[0]]: str(v) for k, v in constants.items()})
        try:
            compiled = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
                                      target=target, options=options)
        except Exception as error:
            assert failure_expected, (name, error)
            expected = "numel (2097152) exceeds triton maximum tensor numel (1048576)"
            assert expected in str(error), (name, error)
            log = out / (name + ".error.txt")
            log.write_text(traceback.format_exc())
            row.update(status="expected_frontend_rejection", exception=type(error).__name__,
                       error=str(error), artifact=dict(path=str(log), sha256=sha(log)))
        else:
            assert not failure_expected, name
            ptx = out / (name + ".ptx")
            ptx.write_text(compiled.asm["ptx"])
            assert "ld.global" in compiled.asm["ptx"] and "st.global" in compiled.asm["ptx"]
            row.update(status="control_compiled", artifact=dict(path=str(ptx), sha256=sha(ptx)))
        records.append(row)
        print(name, row["status"], flush=True)

report = dict(status="CONFIRMED_SOURCE_DOMAIN_DEFECT", triton=triton.__version__,
              torch=torch.__version__, gpu_available=torch.cuda.is_available(),
              gpu_execution=False, admission="unchanged checks with CPU storage/CUDA facade",
              compiler="actual JIT binder, original source, default options, offline CUDA SM80/SM90",
              source=dict(path=str(source), sha256=sha(source)),
              original=dict(path=str(original), sha256=sha(original)),
              compiler_block_limit=TRITON_MAX_TENSOR_NUMEL,
              independent_target="For X=1, cosine=1, sine=0 and scalar offset=0, every output is 1.",
              results=records)
path = out / "report.json"
path.write_text(json.dumps(report, indent=2, default=str) + "\n")
print("PASS: four admitted frontend failures and four positive compiler controls; no GPU execution.")
```
