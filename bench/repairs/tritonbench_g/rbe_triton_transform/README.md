# `rbe_triton_transform`: selected FP32 kernel

The selected source is `rbe_triton_transform_repaired.py`, SHA256
`9095bec159a1bceccea0b6415d198eb1efed6fd30947e79d378ff6f70f17c592`.
Python, guards and the canonical ledger are unchanged. Historical host proofs
remain optional in `Proofs/HostCorrect.lean` and `Proofs/HostFPEquiv.lean`;
`README.host-history.md` preserves their original documentation.

For even pair index `n = 2 floor(c/2)`, the independent real specification uses
`angle = (position + row) / 10000^((n % 128)/128)` and rotates `(u,v)` to
`(u cos(angle) - v sin(angle), u sin(angle) + v cos(angle))`.
A missing final imaginary partner is zero; the real tail is still written.

The public endpoints are `Correct.kernel_correct` and
`FPEquiv.kernel_native_equivalence`. They run the entire selected DSL once per
program against memory left by its predecessors, for any permutation of the
complete grid. The same shared `Torch.FP32.Tensor` supplies metadata, physical
indices, backing storage, offset and pointer. The results cover every output,
complete region frames, concrete little-endian byte stores, global native byte
equality, and preservation of the entire input backing allocation.

`Selected.typed` retains the source's integer widenings, frequency FP32 cast,
trigonometric nodes, barriers and distinct real/imaginary masks. Its task-local
quotation supplies nonnegative index-carrier annotations and keeps signed token
values as `Int`. It lowers the visible int64 widenings of Nat metadata, PIDs
and aranges to identities on the Nat carrier; explicit casts of signed token
values remain Int. `Selected.carrier_types` and `Selected.store_types` check the
expanded register and store tags by definitional equality. `KernelIndex`
relates these mathematical indices to actual
32-bit PIDs widened to 64 bits, signed division/remainder, control arithmetic,
all scheduled token positions, masks, active pointers, alignment and byte spans.
The explicit `tl.toReal` notation exposes the core's integer-to-floating lift;
it does not introduce an extra numerical rounding operation. The adapter also
reifies the implicit FP32 pointer-element conversion on each output store; the
shared store constructor otherwise retains the arithmetic value's Real tag.

The native target is independent of DSL execution. Scalar conversion, division,
pow, sin, cos and FMA remain explicit local primitive bindings. The selected
lowering profile multiplies trig-first, fuses the left product of subtraction
and the right product of addition, and preserves the other rounded product.
These ordinary primitive/compiler bindings are model boundaries. There are no
new numerical deferrals or per-task numerical experiments. The real theorem
is the arithmetic idealization; it does not assert unconditional native FP=Real.

The complete guarded data domain includes empty shapes, nonzero storage offsets,
unused singleton strides, odd columns, all FP32 bit patterns, Boolean Python
positions and signed-64 positions whose padded row positions fit. Tensor/API
and fresh output allocation laws are ordinary entry bindings, with no result
or execution premises. Empty requests run no programs and impose no launch cap.

Native launch support is separate from guard admission. The unchanged guards
accept shape `(1,131071,1)`, position zero, whose grid is `(1,65536)`; standard
CUDA grid-y support stops at 65535. `Index.SupportedLaunch` states this ordinary
native launcher boundary explicitly. No host launch repair or grid splitting
is claimed. The original program also has a concrete stride-two defect:
backing `[1,2,3,4]`, shape `(1,1,2)`, last stride 2 logically pairs `(1,3)` but
loads `(1,2)` because its imaginary neighbor is `+1`.

`check_kernel_source.py` checks the two Python pins and AST formula, masks,
casts and stride neighbors, plus exact integer boundary examples. It performs
no floating-point comparison and launches no GPU. Lean compilation uses the
authorized workspace drivers `/tmp/veritile_workspace_lean.py` and
`.veritile-cache/build-tools/workspace_lean_local_first.py`. The latter checks
disjoint local-package/toolchain namespaces before prioritizing their lookup;
workspace overrides still precede the baseline, with the same artifact selection.
Both drivers, final source/artifact pins and the joint axiom audit are recorded
in the task delivery packet.
