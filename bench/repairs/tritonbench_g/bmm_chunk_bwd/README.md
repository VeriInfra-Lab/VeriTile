# FP32 chunked BMM backward kernel

For every valid output coordinate `[b,s,g,n]`, put `c=s//C`, `r=s%C` and
`L=min(C,S-c*C)`. The independent mathematical result is

```
sum(Dout[b,c,g,k,r] * A[b,c*C+k,g,n], k=0..L-1)
  + (Residual[b,s,g,n] if residual is present else 0).
```

`Target.lean` defines this formula over arbitrary Real arrays, independently of
kernel execution, grid tiling and output memory. Real is the mathematical
interpretation; native TF32 instructions are not asserted to equal exact Reals.

## Source and precision

The selected Python source is `bmm_chunk_bwd_repaired.py`, SHA256
`91e5b8d4d3d423bb1a00fadb347273c5d90d2a3afd5d2b05b51f98eadd90c737`.
It is unchanged in this revision. The original benchmark and its historical
source-defect diagnosis remain separate.

`Kernels.lean` specializes the single readable DSL template shared with
`bench/tritonbench_g/bmm_chunk_bwd/Kernels.lean` to the selected FP32 program.
`Kernels.fp32Source` uses the original head-index expression; the candidate
`Kernels.indexCommuted` changes only `pid_ch - pid_c * ngroups` to
`pid_ch - ngroups * pid_c`. The original specialization is proved equal to the
existing source used by the retained proof library. No parallel handwritten AST
is introduced. Historical storage variants are reused only as proof library
specializations, not added to the current FP32 input domain.

Inputs, residual, accumulator and output retain FP32. The literal source
`acc += tl.dot(dout, a)` retains operand order and two-argument dot syntax.
The observed Triton 3.7.1 SM90 compiler profile stages its dot and following add
into live-carry K8 instructions: configurations 0–5 use WGMMA64×N×8 and 6–8 use
MMA16×8×8. This is an explicit compiler model, not a theorem equating separately
rounded dot/add with carried dot. Native compiler selection depends on structure
and configuration, never input-word payloads.

## Input contract

Python and `Contract.lean` agree on the following conditions.

- A is FP32 `[B,S,K]` or `[B,S,G,K]`. Dout is matching FP32
  `[B,ceil(S/C),C,C]` or `[B,ceil(S/C),G,C,C]`, with `C>0`.
- Optional residual and supplied output have exactly A's shape and FP32 dtype.
- Used tensors are strided and contiguous, on one CUDA device, have no unresolved
  negative view flag, and occupy pairwise disjoint data spans.
- Every used tensor has last queried stride 1, including empty tensors. All
  queried strides of nonempty tensors equal the literal canonical strides.
  Contiguous singleton views with different queried strides are explicitly
  rejected by the existing contract.
- Byte extents are below `2^63`; nonempty pointers are aligned to four bytes.
  For nonempty A, grid y and z are below `2^16` and every configuration's grid x
  is below `2^31`. No grid-product restriction or payload-value filter is added.
- Empty dimensions, including K=0, have no kernel programs and no output cells.

The existing intentional `.contiguous()` source branches are inactive under
these checks. The kernel receives admitted buffers directly. No new copies,
conversions or general compatibility path is added.

## Public proof scope

`IO.lean` exposes three finite input channels (Dout, A and optional residual),
the full output channel, original strides and masks, and the exact three-axis
source schedule. The last chunk keeps the source's distinct Dout-row and output
masks. Inputs may alias each other; the geometric output-separation conditions
remain explicit. There is no global scratch. `PublicIO.mathematical` is the
independent finite sum over logical inputs.

`Correct.bmm_chunk_bwd_real` uses the shared
`PreparedAlgorithmProgramIO.Implements` judgment: successful complete original
source execution, every active mathematical output, exact exterior frame, and
proved access safety along every actual program predecessor when declared IO
windows are in bounds. Each CTA retains the original fresh-register/zero-undef
convention and all three grid dimensions. No allocation, host or Tensor
execution is part of this main theorem.

`FPEquiv.bmm_chunk_bwd_equiv` is the standard `FP.Program` equivalence for the
entire original and candidate quoted programs. It has an empty transformation
assumption list and preserves arbitrary original floating algebras and states.
It changes only an exact integer product. The ordinary
`KernelInterface.versions_eq` theorem composes that very same source result over
the complete original grid; the per-program statement alone is not a claim of
whole-grid output correctness.

The original flattened Real entry is retained verbatim as the ordinary theorem
`Proofs/KernelRealResult.lean:bmm_chunk_bwd_real`.
`PublicIO.flatten_execution` explicitly connects the public finite IO schedule
to that same original flattened execution. Both reuse `HostGrid.run_correct`
and `HostGrid.run_flatten`; no second execution proof is maintained.
`Proofs/KernelNativeResult.lean` retains the original `native` export, whose
`KernelInterface.native` foundation still proves complete source/candidate
success, `FPFlat.lift/publish`, actual `NativeExecution.grid`, all output
recurrences, `Good`, and the full physical exterior frame. All nine
configurations, original arbitrary aliases/placement domain and the SM90
staging/TF32 K8 primitive model are unchanged. This full native transport is a
named supporting theorem, distinct from the public source-equivalence scope.

`TensorBridge.bmm_chunk_bwd_real`, `TensorBridge.bmm_chunk_bwd_equiv`,
`TensorBridge.outputs` and `TensorBridge.native_frame` preserve all original
Tensor statements and proofs. Their original logical-input bindings, aligned
actual addresses, four-byte observations and physical frames remain intact.
Both interfaces share `HostGrid.flat_correct`, `NativeExecution.source_grid`
and `FPGridOutput.complete`, with the unchanged Direct layout/admission adapters.
`Misc/Review.lean` collects cross-interface audits. Main import closures contain no
TensorBridge, and supporting proofs import no public reader entry.

Only targeted Lean builds were run for this formatting change. No new numerical
assumption, GPU experiment, floating rewrite, independent cold replay, or
runtime/compiler compatibility project was added.

## Source-only cleanup (2026-10-21)

The selected FP32 original and existing integer-index candidate now use complete
ordinary quotations, `Kernels.quotedOriginal` and `Kernels.quotedIndexCommuted`.
`Kernels.bindMain` retains the same scalar arguments, all nine tunings, residual
and output precision, masks, branches, dot order and stores.
`Proofs/SourcePreservation.bindMain` proves equality of the complete function
for both versions to the archived previous source. Existing public proof names,
contracts, numerical dependencies and completion status are unchanged.
