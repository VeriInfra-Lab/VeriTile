# matmul_dequant_int4: exact grid-y contract candidate

This separate candidate adds exactly the missing guard `(N + 31) // 32 <= 65535` in `_packed`, after the existing formed-index checks. It applies to standalone dequantization and the dequantization stage used by production `s1`. The original source, its 44 source/document files, and its frozen defect packet remain unchanged. Canonical selection belongs to the root reviewer.

The accepted domain is exactly the original domain intersected with `N <= 2,097,120`. `Contract.packed_domain_exact` proves equivalence to satisfying the y-axis limit for all eight original configurations, including duplicate autotune entries. This is the weakest common condition because six configurations have BN=32. No extra product, dtype, finiteness, K, group, stride or layout restriction is added. All empty shapes were rejected already and remain rejected. Original x-axis and pointer bounds are retained and derived in `Contract.launch_axes` and `Proofs/Safety.lean`.

`Misc/Whole.lean` imports the complete task proof. `DSL.lean` displays both complete readable shared `triton` quotations with definitional binding proofs. There is no numerical transformation between source versions. Both original Python JIT bodies, all 8+17 configurations, both `NO_GROUPS` branches, the quantizer and Torch calls are unchanged.

The standalone proof covers both FP16 and FP32 profiles, every program and active output, and the full typed memory frame. `TensorBridge.real` proves the independent base-16 Real formula at actual Tensor coordinates:

`D[k,n] = (digit(Q[k//8,n],k%8) - digit(Z[k//G,n//8],n%8)) * S[k//G,n]`.

`TensorBridge.native` proves the selected-width signed-conversion/multiply result for every packed word and every floating payload. A 32-bit carrier holds FP32 words or the low 16 bits of an FP16 word. The explicit promotion in the shared quotation uses one finite conversion port for the 31 possible FP16 nibble differences; FP32 needs no such port. Operations remain named native primitive interfaces, so these theorems do not claim experimental or universal compiler conformance. `Tensor.Actual` binds shared Torch metadata, actual storage identity and physical indices; `Tensor.Post` and `Frames.inputs` preserve every input allocation and every output exterior cell. No allocator implementation, host-return equality or unified byteheap outcome is a theorem premise.

The retained raw fused kernel keeps its original explicit raw ABI: full K/group multiples of 128, N multiple of 256, canonical strides and FP16 inputs/output with FP32 accumulation. `Raw.valid` derives the complete-source preconditions from that original contract. It does not require the sibling task's stronger host/index checks. `RawTensor.real` proves the independent dense Real product and actual Tensor frame; `RawTensor.native` proves the ordered fused native target, output bytes and full allocation frame using the already established scalar and fixed-K16 ports. These raw restrictions do not apply to standalone dequantization.

Production `s1` launches only the dequantization JIT. `S1.kernel_real` and `S1.kernel_native` connect its unchanged accepted input domain to that complete kernel proof. The quantizer and `torch.mm` are ordinary Torch operations, whose existing helper/library and host proofs remain imported under the original namespace. They are not new conditions for the kernel endpoints.

Validation: 24 actual guard/wrapper AST boundary/profile/stride cases and 12 empty-shape cases pass. The predecessor's 34 actual Triton interpreter cases and six offline JIT-binder SM80 compilations are reused because the JIT ASTs are identical. No GPU execution is available. Current task numerical transformations and required numerical experiments: zero. See the frozen candidate delivery for current source, artifact, evidence and import-closure hashes.

`Correct.lean` exposes `Correct.kernel_real`; `Misc/NativeCorrect.lean` retains
the ordinary `FPEquiv.kernel_native` theorem. Both use the shared
`ProgramIO.Implements` judgment over logical packed-word/scale inputs and an
independent nibble-difference times scale target. `IO.lean` lists the signed
INT32 read windows, selected FP16/FP32 channels, exact source strides and
complete two-axis grid. `Proofs/AccessSafety.lean` follows the actual source
address/mask states; `Proofs/PublicIO.lean` connects those bounds and the
existing complete `KernelGrid.whole` result to the shared IO relation.
Both profiles, all original payloads and the existing finite conversion port
are retained. Old raw results live in `Proofs/RealResults.lean` and
`Proofs/NativeResults.lean`. The complete original Tensor/physical-frame
statements keep their names in `Proofs/TensorResults.lean`, exposed through
`Misc/TensorBridge.lean`; the S1 and frame helpers import that foundation directly.
`Misc/Whole.lean` still includes both the standalone and retained raw fused kernels.
No Python, kernel quotation or input contract changed in this format update.

The 2026-10-21 implementation pair compares that complete standalone source
with `unitSource`, an actual contiguous-output specialization. Its `fpb_offs`
changes from `offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn`
to `offs_k[:, None] * stride_fpbk + offs_n[None, :]`. The existing
`KernelGrid.Valid.outputColumn` already requires `stride_fpbn = 1`, so no input
restriction is added. This eliminates a runtime vector integer multiplication;
no measured speedup is claimed. `matmul_dequant_int4_unit_stride.py` supplies a
complete callable candidate, with all eight standalone configurations, fused
source, quantizer and wrapper checks retained.

`FPEquiv.unit_stride_equiv` states `original f M p ≡[[]] unitStride f M p` for
both source profiles and any typed arithmetic backend `M`, on exactly the existing
valid layout. The proof equates the changed pure assignment (including missing
register failures), full kernel states, trace safety, and the complete original
grid. The shared public IO supplies full outputs and exterior typed-cell frames.
Every FP operation, signed nibble conversion, source precision and order is
unchanged, so the main pair needs no numerical law or optional finite conversion
port. The retained optional native formula keeps its existing conversion premise.
No numerical experiment is deferred.

Optional application proofs and readers are under `Misc/`; their declaration namespaces and theorem statements are unchanged. Required main proof helpers remain under `Proofs/`.
