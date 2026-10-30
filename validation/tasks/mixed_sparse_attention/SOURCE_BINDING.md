# Source and execution binding

Original benchmark Python is unchanged:
`cadf55fd56b6ac21b74cf88501ca43b90a685db901bdaad2312a1a3f42933c0e`.
The selected repaired JIT AST digest (`ast.dump`, without location attributes)
is `f5eaff450d5cb5008e79e9e564b320284f34f108a048a48396ec33e332922a35`.
`check_contract.py` checks this digest. The checked wrapper is outside the JIT
AST and adds admission and fresh-allocation checks.

The typed source under `Proofs/Whole002/Kernels.lean` retains the selected
FP16 specialization, both sparse loops, source order, guarded reductions,
all casts, and stores. The explicit final FP16-to-FP32 widening represents
Triton's implicit store coercion to the FP32 output pointer. The saved BF16
source syntax is a proof helper; the wrapper admits only FP16 computation.
`Source.kernel_exact` proves equality to the parsed typed kernel.
`Source.whole_typed`, `source_memory`, whole-grid execution, fresh allocation,
and `Physical.invoke` establish complete typed-source execution.

The current public source adapters actually execute `Physical.invoke` and
extract its returned cells. Failure and out-of-range extraction branches are
proved unreachable on admitted inputs. The Raw dispatch performs the same
validate/empty/live branch and returns the supplied fresh allocation's raw
descriptor. Its address function uses that descriptor's actual pointer.
`Reify.Layout` only names argument roles with symbolic regions; each cell is
constructed from its current raw word. Distinct symbolic roles may have
identical or overlapping physical read-only pointers. No result relation is
contained in this layout. The checked access theorem proves all enabled
source loads use those current addresses.

## Ordinary realization, separate from numerics

As with other typed-source models, Python/PyTorch descriptor observation,
FP32 scalar marshalling, successful fresh allocation, typed memory ABI and
CUDA execution are external runtime realization obligations. The Real model
uses the finite decoded ABI scale word and ideal finite input interpretation.
The native model consumes the exact `Request.scaleWord` and raw input words.
The local memory API and little-endian codec describe ordinary reads/writes;
they contain no attention or output-value equation.

The selected native backend explicitly consumes scalar operation stages,
FP16 narrowing and widening, fixed m16n8k16 instructions, the selected
four-owner reduction tree (sequential local pairs, XOR2 then XOR1), and the
physical output owner. These are executable parts of the source interpretation,
not unused compiler-plan indices. Correspondence of this interpretation to
an emitted Triton/CUDA program remains an external compiler realization
obligation, including operand/fragment routing, reduction order, cast and
store order, and enabled instruction/control correspondence. The offline
SM80 profiles are evidence for this selection, not a universal compiler proof.
No endpoint assumes a whole attention result or a whole reduction equation.

## Local numerical obligations

`Native.ComparisonTransport id actual standard` contains only transport of
raw words, literals, unary, binary, ternary, one coordinate of a fixed
m16n8k16 instruction, and the local scalar comparison operation. The recursive
graph theorem derives whole native value equality from these local laws.
There is no whole-dot, reduction, online-softmax-step, or output port.
No owner-uniformity, max commutativity or addition commutativity premise is
needed by the public endpoint; it reads the actual selected physical owner.

The numerical interpretations must account for exceptional FP32 words,
signed zero, subnormals/FTZ, exp2 approximation, division, fused operations,
and FP16 conversion/MMA behavior. Input payloads are not restricted to finite
words by the wrapper. The guarded all-masked and zero-normalizer paths are
retained in the recurrence; this is not an accuracy theorem against Real
attention. GPU numerical execution was unavailable in this environment.

## Exact Tensor observations and returned logical values

The local descriptor is an observation of an already classified Tensor:
`isinstance(t, torch.Tensor)` is checked before its header is inspected. The
selected `floating` Boolean is exactly FP32 for Q/K/V/output and exactly
int32 for metadata; it does not stand for an arbitrary floating dtype.
`_validate` checks these exact dtypes and `t.layout == torch.strided`, shape,
device, contiguous layout, lazy-negative/conjugate flags and inference status.
The local descriptor fields correspond to these API observations:

| Local field | Python observation |
| --- | --- |
| `object` | Tensor object identity |
| `shape`, `strides` | `t.shape`, `t.stride()` |
| `floating` | selected `torch.float32` / `torch.int32` classification |
| `strided`, `cuda`, `device` | `t.layout == torch.strided`, `t.is_cuda`, resolved device index |
| `contiguous`, `negative`, `conjugate` | `is_contiguous()`, `is_neg()`, `is_conj()` |
| `requiresGrad` | `requires_grad` |
| `pointer` | `data_ptr()` |
| `storageBase`, `storageBytes`, `storageOffset` | `untyped_storage().data_ptr()`, `.nbytes()`, `storage_offset()` |

`Schema.header` consumes these selected observations; its four stride ABI
fields equate source parameters with the same observed strides, from which
`Admission.contiguous` and `Admission.shape` derive the source layout.
Successful `torch.zeros_like(q)` supplies the actual fresh output descriptor;
`_output_storage` checks its entire extent and separation from input backing.

`Proofs/TensorBridge.lean` makes the returned allocation bridge explicit using
shared `Torch.FP32.Tensor` and `TensorLaws`. `Observation` identifies object,
shape, dtype, strided/device/view flags, actual pointer and backing allocation.
Its premises contain no execution or output-value relation. `logical_read`
uses checked shape/strides, shared element bounds/contiguity, and the same
allocation pointer to derive the raw-to-logical read equation. The public
`Checked.FP.checked_tensor_call` consumes this bridge to conclude actual
returned Tensor identity/shape, logical FP32 values, empty behavior, and frame.
The stronger raw endpoint separately exposes all four bytes and unchanged
signed metadata of that same returned memory.
