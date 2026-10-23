# Kernel-only delivery for the selected FP32 repair

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`.

Current source-pair review (2026-10-21): The current `FPEquiv.lean` has two actual complete-source pairs: recurrent carry/dot addition versus the existing sum-left candidate, and prefix addressing versus the existing index-first candidate. `PairIO.lean` specializes the retained repair candidates to exactly the selected FP32 source/region roles and every original launch base. The recurrent pair retains its reached finite-operand sites and uses only the shared strict FP32 ADD-COMMUTE certificate; prefix has an empty domain/rule list. Actual experimental rules used: zero. The old unrestricted full-grid `inner_fp_equiv`/`prefix_fp_equiv` declarations remain ordinary theorems with unchanged names/statements/proofs in `Misc/NativeCorrect.lean`. Those native grid observations are auxiliary; this change does not claim a new native whole-grid pair connection.

This package proves the two kernels in the already selected
`../chunk_gated_attention_repaired.py`. It changes neither that Python source
nor its admitted domain. The historical host proofs and their four standard
files remain intact. The current kernel entries are `Contract.lean`,
`Kernels.lean`, `Correct.lean`, and `FPEquiv.lean`. The main correctness and
native interfaces use shared IO judgments over finite logical channels and
kernel geometry. Raw execution states and per-address proof details stay in
supporting modules.
`Misc/TensorBridge.lean` retains the complete checked Tensor applications, supported
by `Tensor.lean` and `Memory.lean`.

The original benchmark remains separately classified as a source defect.
`bench/tritonbench_g/chunk_gated_attention/Proofs/UndefinedTail.lean` gives a
full-source counterexample: with T=1 and 16-wide dot tiles, valid inputs zero
and unspecified inactive K/V values one, the source may store 15 instead of
the zero-extended formula's 0. This is a semantics counterexample, not a claim
that a GPU was observed choosing those unspecified values. The present
delivery concerns the previously accepted repair, not a new repair candidate.

## Target and contract

`prefix_correct` proves the independent within-chunk sum

\[
O[h,t,f] = \sum_{u=\lfloor t/BT\rfloor BT}^{t} G[h,u,f].
\]

`inner_correct` treats its G input as an independently supplied cumulative gate
tensor. Let e(c)=min(c BT+BT-1,T-1), and let f=k in gateK mode or f=v in gateV
mode. The target has H₀=seed or zero, stores history[c]=H_c before updating,
and has

\[
H_{c+1}[k,v]=H_c[k,v]e^{G[e(c),f]}+
\sum_{0\leq j<BT,\ cBT+j<T}
 K[cBT+j,k]V[cBT+j,v]e^{G[e(c),f]-G[cBT+j,f]}.
\]

The formal target is the independent closed form `Host.innerState`, with the
source's gateK/gateV multiplication association retained. Optional final output
is H_NT. Prefix and inner remain separate APIs.

The unchanged Python checks admit contiguous, ordinary FP32 CUDA tensors on
one device, with no unresolved negative/conjugate flag; natural dimensions,
power-of-two blocks at least 16, the existing tile-element limit, checked
signed-64 byte extents, and aligned nonwrapping pointers. Read inputs may alias
and may have nonzero storage offsets. `Bindings` connects the same observed
descriptors to `VeriTile.Torch.FP32.Tensor`, `TensorLaws`, backing storage and
logical elements. It does not invent a new set of metadata booleans.

The existing source stages incomplete tiles. The Tensor bridge therefore takes
the existing `InnerPrepared`/`PrefixPrepared` ordinary copy/zero-fill/repeated-
gate API relations. Full tiles retain their original inputs. These relations
specify only input bytes, shapes, direct-input identity, allocation separation
and copy frames; none supplies a dot, gate arithmetic, kernel result, successful
execution, or lowering result. They are an explicit preprocessing API boundary,
not a new input restriction or a proof of Torch's implementation.

The bridge conclusions concern the actual prepared kernel output Tensor and
the valid logical coordinates of the caller problem, via the proved offsets
into its padded shape. They stop before the existing output slice/copyback.
`active` is the exact existing nonempty launch branch. Empty inputs skip the
kernels; `inner_empty_history` and `prefix_empty_output` state their zero output
counts. The empty host's optional zero-plus-seed/copyback behavior remains in
the retained historical host result and is not a new kernel proof obligation.

## Proof and precision

- The shared readable source is the existing `NativeSource.cumOriginal` and
  `NativeSource.hOriginal` in `../Kernels.lean`. `inner_real_source` and
  `prefix_real_source` connect the same FP32 source selections. No second
  handwritten kernel AST or dtype variant is introduced.
- `RealIO.lean` declares the shared `AlgorithmProgramIO` interface: the actual
  source-derived projection, complete original ordinary-grid stages, input
  roles, logical outputs and explicit padded scratch. `inner_correct` and
  `prefix_correct` state `ImplementsOn` with independent mathematical formulas
  and finite input materialization domains. The existing zero-undef selection
  is applied once before those stages. Actual source access safety, successful
  complete execution, all valid outputs and the exterior frame are conclusions.
- `NativeIO.lean` declares shared `ProgramIO` channels using the finite actual
  read/write windows of every original program. The recurrent input tags are
  Real for k/v/g and FP32 for h0; history stores are Real with the explicit
  FP32-profile cast, while final and prefix stores are FP32. Input binding uses
  the original decoded-read behavior, with arbitrary symbolic graph operands
  and arbitrary initial stored tags. No new input-value guard is imposed.
- `inner_fp_equiv` and `prefix_fp_equiv` use shared `ProgramIO.Observes` against
  independent native arithmetic targets. The complete source schedule includes
  all ordinary-grid lanes and prefix dummies. `Proofs/NativeMerge.lean` proves
  its final memory exactly equals the original simultaneous per-grid merge
  and successive batch execution, for every original native config, including
  BT=0 and arbitrary BS. All original scalar observations and exact raw frames
  remain available in `Proofs/RawFP.lean`; original Real endpoints are retained
  in `Proofs/RawReal.lean`.
- The same four names under `TensorBridge` retain the original actual Tensor
  conclusions, full prepared-buffer domain, signed-pointer/input bindings,
  four-byte native serialization and exterior byte frames. They reuse the
  existing raw proofs and checked-to-memory correspondence.
- Native arithmetic retains FP32 loads, explicit casts, zero-plus-seed,
  history-before-update stores, live gated dot accumulators, the ordered
  scalar FMA chain, and exp lowering through word `0x3FB8AA3B` followed by exp2.
  Real is the ideal mathematical interpretation; it does not assert that
  rounded native words equal exact Real values.

The native primitive operations and their pointwise realization laws are an
ISA-model boundary. `ScalarLaws` has only fixed local literal, binary, unary,
cast and FMA laws; raw bit leaves are the identity. There is no whole-dot,
whole-tensor, whole-execution or whole-lowering premise. These local laws are
not empirical numerical assumptions. The delivery makes no approximation-error
claim, so it adds **no deferred numerical experiment assumption**.

Compiler realization of the selected scalar instruction order, resource
availability, and the ordinary Torch/storage API implementation remain external
boundaries. The offline evidence is the existing narrow Triton 3.7.1 SM80
profile: both gate branches with BT/BK/BV16, two chunks, seed/final enabled,
two warps, plus prefix BT/BS16. It observes live dot carry, zero-plus-seed,
`fma.rn.f32` and `ex2.approx`; it is neither a compiler proof nor coverage of
every allowed block size, autotune configuration or GPU architecture. The Real
source theorem covers all three prefix BS choices; this must not be confused
with the narrower compiled-profile observation.

Model references:

- [Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
  documents unspecified out-of-bounds block-pointer values when no padding
  option is supplied; the selected repair makes active source tiles complete.
- [Triton dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
  documents dot accumulation and IEEE/TF32 input-precision selection.
- [Triton exp](https://triton-lang.org/main/python-api/generated/triton.language.exp.html)
  documents the approximate exponential primitive.
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
  specifies natural alignment for memory operations, `fma.rn.f32`, and
  `ex2.approx.f32`. The actual pinned compiler artifacts supply the local
  instruction-profile evidence; the docs alone do not prove compiler routing.
- [Tensor data_ptr](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html),
  [storage_offset](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html),
  [is_contiguous](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html),
  and [resolve_neg](https://docs.pytorch.org/docs/stable/generated/torch.resolve_neg.html)
  are the API basis of the shared storage binding.

## Validation and preservation

The original six-module delivery was built with the existing v2 workspace driver;
public axiom checks use only the standard base and mathematical/native targets
pass non-circularity checks. The existing CPU control script passed 19 preserved
floating ASTs, all nine retained autotune declarations, 45 rejected metadata
cases and 24 exact controls. These controls exercise the actual source in the
CPU interpreter with CUDA admission/context adapted; they are not GPU tests.
The existing three offline SM80 profiles were recompiled. No CUDA GPU was
available. Independent replay/review is tracked separately by the root agent.

The original kernel-only delivery did not alter the ledger, Lake, shared core,
original Python, selected repair Python, or the 122 baseline task files. Its
hashes, build receipts, dependency-cache boundaries and compiler artifacts
remain frozen under
`.veritile-cache/kernel-evidence/chunk_gated_attention/`.

The strict interface split was validated separately under
`.veritile-cache/kernel-interface-split-20261020/batch07/`. Its primary files
do not import `TensorBridge`; the existing scalar realization assumptions,
FP32 source precision and Tensor-supported domain are unchanged.

The public IO-format follow-up is recorded under
`.veritile-cache/public-io-format-20261020/` (batch13). The two main files have
strict statement-surface audits; task-root `../Misc/InterfaceChecks.lean` checks
both distinct kernels, the retained raw endpoints, actual source safety,
ordinary-grid/schedule memory equality, and all four unchanged Tensor endpoints.
This follow-up adds no numerical experiments, GPU execution or independent
replay claim, and leaves Python, both Kernels files and Contract unchanged.

Source readability maintenance (2026-10-22): the shared `../Kernels.lean` now
contains complete ordinary Triton quotations for both prefix orders and every
retained recurrent storage/order specialization. `SourceDefinitions.lean` selects
those source definitions directly; the unchanged Real/source connection theorems
are in `Proofs/SourceBridge.lean`, re-exported by `Kernels.lean`. Complete function
equality to the previous macro expansions and selected source aliases is checked
in the source-only campaign cache. Programs, domains, precision and main pairs are
unchanged.
