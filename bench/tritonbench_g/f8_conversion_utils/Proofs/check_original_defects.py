"""Execute the original wrapper AST with storage metadata; no GPU/numeric test."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "f8_conversion_utils.py"
WIDTH = {"int8": 1, "fp16": 2, "fp32": 4}


class Tensor:
    def __init__(self, *, count, dtype, storage_count, offset=0, stride=1):
        self.count, self.dtype, self.storage_count = count, dtype, storage_count
        self.offset, self.stride, self.device = offset, stride, "cuda:0"

    def numel(self):
        return self.count

    def element_size(self):
        return WIDTH[self.dtype]

    def untyped_storage(self):
        return SimpleNamespace(size=lambda: self.storage_count * self.element_size())


calls = []


class Kernel:
    def __getitem__(self, grid):
        def launch(y, x, count, **meta):
            calls.append(dict(input=x, output=y, count=count, grid=grid(meta)))
        return launch


torch = SimpleNamespace(Tensor=Tensor, int8="int8", float16="fp16", float32="fp32",
    empty_like=lambda x, dtype: Tensor(count=x.count, dtype=dtype, storage_count=x.count))
triton = SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b, reinterpret=lambda x, _: x)
tl = SimpleNamespace(float8e5="e5")
tree = ast.parse(SOURCE.read_text())
wrappers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("f16_to_f8", "f8_to_f16")]
environment = dict(torch=torch, triton=triton, tl=tl,
    kernel_f8_to_f16=Kernel(), kernel_f16_to_f8=Kernel())
exec(compile(ast.Module(body=wrappers, type_ignores=[]), str(SOURCE), "exec"), environment)

# backing = torch.empty(8, dtype=torch.float32, device="cuda"); x = backing[2:5]
slice_input = Tensor(count=3, dtype="fp32", storage_count=8, offset=2)
environment["f16_to_f8"](slice_input)
slice_call = calls.pop()
assert slice_call["count"] == 8 and slice_call["output"].count == 3
assert slice_call["grid"] == (1,)
assert 3 < slice_call["count"] and not 3 < slice_call["output"].count
# Offsets six and seven also read past the original input backing allocation.
assert slice_input.offset + 6 == slice_input.storage_count

# backing[0:6:2] passes the dtype/device assertions. Flat lane one reads
# backing[1], while logical input coordinate one is backing[2].
strided_input = Tensor(count=3, dtype="fp32", storage_count=6, stride=2)
environment["f16_to_f8"](strided_input)
stride_call = calls.pop()
assert stride_call["count"] == 6
assert strided_input.offset + 1 != strided_input.offset + 1*strided_input.stride

# Decoder N is output storage size, so a stepped byte input gives a logical
# mismatch without relying on the encoder's input-storage-count error.
decoder_input = Tensor(count=3, dtype="int8", storage_count=6, stride=2)
environment["f8_to_f16"](decoder_input)
decode_call = calls.pop()
assert decode_call["count"] == 3
assert decoder_input.offset + 1 != decoder_input.offset + 1*decoder_input.stride

print(json.dumps(dict(source=str(SOURCE), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    original_assertions_admit_all_cases=True,
    cases=[dict(name="encoder_offset_slice", logical_count=3, storage_count=8,
                passed_N=8, first_output_overrun_lane=3, first_input_overrun_lane=6),
           dict(name="encoder_stride_two", passed_N=6, logical_index=1,
                expected_storage_element=2, actual_storage_element=1),
           dict(name="decoder_stride_two", passed_N=3, logical_index=1,
                expected_storage_element=2, actual_storage_element=1)],
    gpu_execution=False, numerical_experiment=False), indent=2))
