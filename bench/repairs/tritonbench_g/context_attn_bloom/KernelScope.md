# Kernel revision: context_attn_bloom

The selected source is `context_attn_bloom_repaired.py`, SHA256
`9b80a67487612a00769133e0794cea1f39b61ee23cbf02fedbaf61d661d7faee`.
The historical source, guard, contracts, and whole-call results remain intact.
This revision adds kernel entry points and fixes the native profile used by
those entry points. It does not add host obligations.
`Whole001/FPCore.lean` gains one compatibility import of the existing
`context_attn_fwd` floating execution module. This lets the shared interpreter
equations enter the environment before simplification exports them again;
all existing definitions and theorem statements are unchanged.

## Independent mathematical target

`Checked.mathematical` is the existing independent prefix-sum formula in
`Proofs/CheckedReal.lean`, with its recurrence in the original
`Proofs/NormalizedFormula.lean`. It uses natural exponential weights, the
finite causal sentinel `-100000000`, zero values for padded key lanes, and the
source's guarded retention of the old normalized accumulator on a wholly
future key tile. The target has no running maximum or source interpreter.
Replacing it with strict causal softmax changes the selected program's
observable behavior. The CPU checker includes both distinguishing cases.

## Precision and admitted inputs

Q, K, V, and output are contiguous FP32 tensors, with grouped query/KV heads,
width 16 through 256, and the source-selected 64 or 128 query/key tile.
Metadata and page tables are contiguous INT32. The existing executable
checks cover shapes, packed intervals, pages, capacities, flags, alignment,
storage windows, output separation, signed indices, grid extents, and formed
pointers. Read tensors may alias. Nonzero storage offsets are allowed.
Empty work leaves the complete physical memory unchanged.

The Real theorem accepts arbitrary Real arrays. The native theorem consumes
raw FP32 words, including NaNs, infinities, and signed zeros. These are separate
interpretations of the same source; no Real-to-FP error bound is asserted.
The launch scalar is the captured ordered host expression `1.0/(D**0.5)`,
with a runtime FP32 ABI signature. The scalar table is checked against IEEE
FP32 packing for all 241 widths; `LaunchScale.Bound` connects that table to
the source parameter. The offline probe confirms the FP32 signature.

## Native boundary

The new entry fixes a payload-independent SM80 profile. Ordered reductions,
producer routing, multiplication orientation, normalizer fusion, and live
K8 TF32 MMA seeds are concrete definitions. No caller-supplied tree selector,
whole dot implementation, output oracle, compiler-binding equation, or
successful-source-run premise appears in the new result.

Numerical interpretation remains at nine local families: literal encoding,
add, subtract, multiply, full division, max, approximate exp2, FMA, and one
output coordinate of the fixed `m16n8k8` TF32 instruction. The TF32 port
includes its documented input truncation/interpretation. In particular it
is not an arbitrary-width dot port. The native graph keeps the source's
constant `0 + -inf`; identifying its compiler-folded `-inf` needs only the
corresponding scalar-add identity at the primitive boundary.

The compiler connection is an explicitly trusted pattern boundary supported
by saved actual-wrapper/native-binder TTIR, TTGIR, LLIR, and PTX artifacts.
Compiler internals and GPU hardware execution are not proved here. The
offline compiler probe does not execute a GPU launcher. Physical output
bytes, tensor readback, untouched output bytes, and all read windows are
proved from the shared source execution.

The compiled `B128/D256` sample requests 327680 bytes of shared memory,
which exceeds the SM80 per-CTA opt-in limit of 163 KiB. Triton's launcher
initialization rejects a kernel whose recorded shared-memory requirement
exceeds the selected device's limit. This profile is a checked instruction
graph artifact, and is excluded from any claim of a runnable SM80 kernel.
The Python guard has no corresponding resource check. Thus this package
does not establish successful SM80 execution over the entire admitted
Python input domain. The selected source and guard are preserved for root
review of this limitation; no new host proof or source repair is added.
