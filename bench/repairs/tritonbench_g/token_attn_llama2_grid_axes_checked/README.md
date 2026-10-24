# token_attn_llama2: checked CUDA grid axes

This separate, initially unselected candidate adds exactly two input bounds
when the existing wrapper launches (`B > 0` and `M > 0`): `Hq <= 65535` and
`ceilDiv(M,32) <= 65535`. Empty calls retain their original admission domain.
Removing the two added Python lines restores the entire preceding repaired
file byte for byte. The complete kernel and launch wrapper are unchanged.

The stage writes gathered QK scores into the caller's supplied output:

```
Out[h, Start[b] + n] =
  sum(d < D, Q[b,h,d] * K[Loc[b,M-Length[b]+n], h/(Hq/Hkv), d]) / sqrt(D)
```

There is no softmax or V reduction. The inherited domain includes FP32
payloads, signed-int32 metadata, D in {16,32,64,128}, positive divisible heads,
one CUDA device, contiguous ordinary views, the original signed32 arithmetic
envelope, valid active suffix page IDs, and disjoint nonempty output segments.
Exact dense strides and physical interval checks remain conditional on a
launch. Repeated page IDs, same-dtype read aliases, and arbitrary signed32
inactive location prefixes remain admitted. Floating values are unrestricted.

`Contract.lean` states successful guard admission in terms of observations of
the original tensors. `checkInput` and its equivalence theorems model successful
outcomes after Python's unchanged Tensor and exact-nonnegative-int
classification; exception ordering and messages are not the specification.
Its head quotient facts are proved equivalent to Python's positive/divisible
head test. `Admission.lean` proves that candidate admission is exactly the
preceding admission conjoined with the two conditional axis bounds, including
an explicit theorem retaining the full preceding empty-call domain.
Metadata values come from signed decoding at the actual observed
int32 pointers. No independent metadata array or assumed output is supplied.

`Correct.lean` states the independent Real sum through the shared
`AddressedProgramIO.ImplementsOn` judgment, using ideal sqrt scaling.
`IO.lean` exposes the full quoted-source schedule, Q/K logical shapes and
arbitrary admitted strides, Length/Start/Loc channels, the active Out mask,
and the exact exterior typed-cell frame. The original `Whole.Valid` domain is
retained; it is not replaced by the narrower dense Tensor specialization.
`Interface.MetadataValues` explicitly binds the logical metadata to the
existing signed data after its exact nonnegative Nat adapter.

`Misc/NativeCorrect.lean` uses the same shared judgment for source-to-native observation,
with address-labelled input words and the independent formula in
`Misc/NativeReference.lean`. The Real target and ordered products/score needed
by the main pair proof are in `Proofs/SourceFormula.lean`; address and finite
input adapters are in `IO.lean`. This is the original result kind, not an equivalence to an
invented second kernel. `Proofs/IOImplements.lean` proves the complete raw graph
before applying the observer. Its source execution and raw frame reuse the
established `NativeGrid.run_correct`; `Proofs/SourceSafety.lean` proves actual
prelude reads, unmasked query loads, masked gathers/stores and the exact 0/1
loop. `Proofs/IOSafety.lean` derives the bounds from declared IO and follows
the actual successor memories through every permitted grid permutation.
TensorBridge retains the complete
applications to original logical Tensor coordinates and physical pointers. The native
reference retains the actual scale words `3e800000`, `3e3504f3`, `3e000000`, and
`3db504f3` for D=16,32,64,128. Primitive rounded add, multiply and FMA meanings
are model parameters. Every reduction plan covers every dimension exactly
once, with optional zero padding; tree and local contraction choices inspect
expression shape. This is not a claim that distinct floating evaluation orders
produce identical values, nor a fixed compiler-profile certificate.

Both TensorBridge application results expose the complete raw typed source execution and frame,
all output cells at actual tensor physical indices, all native caller cells
outside the selected writes, and all input-view cells. The native observation
maps source stores to actual pointer addresses; the raw frame remains public
so that this observation cannot conceal other source stores. Output and read
views may occupy disjoint parts of a shared allocation, so the result does not
claim preservation of every cell of a read tensor's entire allocation.
Empty calls leave caller memory unchanged.

The integer binding proves signed32 exactness from the inherited envelope,
uint64 modular pointer arithmetic, nonwrapping/aligned actual accesses, and
signed decoding of lengths, starts and active page loads. Masked pointer
expressions may wrap; they are not accessed. Payload reads are connected to
`Torch.FP32.Tensor`, its storage/contiguity laws and original logical reads.

The CUDA per-axis bounds use the already frozen official CUDA 12.6.3 Table 21
evidence. Shared launcher signed grid-product compatibility is separately
deferred and is outside this candidate; no product bound is added here.
Kernel semantic success and architectural per-axis admission are the scope,
not an unconditional claim that every installed runtime launch succeeds.

Validation uses the local-first Lean build driver and axiom/noncircularity
audits. `check_candidate.py` checks exact source differences and ten actual
guard/launch-capture cases, including both boundaries, both overflows, both
empty cases and all four dimensions. The capture uses real CPU Torch tensors
with CUDA classification/context facades and a captured launcher. It executes
no GPU kernel and provides no numerical experiment. The original 44 source
and document pins and the frozen defect packet remain unchanged. Root review
alone controls canonical selection and the ledger.

`Misc/TensorBridge.lean` is unchanged and exposes the complete original applications
as ordinary theorems `TensorBridge.Real.token_attention_kernel_correct` and
`TensorBridge.NativeApplication.token_attention_kernel_fp_equiv`, including
both raw and projected native frames, original physical pointers and all input
view cells. It shares the established `Whole.complete` / `NativeGrid.run_correct`
execution foundation with the new IO adapter. The old exact public statements
remain ordinary theorems in `Proofs/RawCorrect.lean` and `Proofs/RawNative.lean`,
including empty raw launch identity without metadata admission. `Correct.lean`
has one specification; `Misc/NativeCorrect.lean` retains one ordinary native theorem.
Both statements keep no operational state or final-memory surface. Neither
imports TensorBridge, directly or indirectly. Python, Kernels, Contract, source
precision, numerical model domains and empty-call admission are unchanged.
Targeted Lean builds and statement/axiom audits validate this organization;
this revision adds no GPU experiment or numerical assumption.

`Proofs/LegacyBinding.lean` proves that the original finite-array initial states
satisfy the new addressed IO binding. `Misc/PhysicalIOBinding.lean` specializes that
result to actual Real/native Tensor memory and proves that the logical Q and
gathered K channels equal the original Tensor reads. It uses the same physical
reader for aliases and adds no copying or read-view separation condition. This
optional module does not occur in either main's dependency closure.

The 2026-10-21 implementation pair adds a complete callable candidate in
`token_attn_llama2_reuse.py` and a full `Kernels.reuseTyped` quotation. It
computes `block_stard_index = start_n * BLOCK_N` once, then uses that scalar
origin in `offs_n`. This removes one repeated integer multiplication; no
hardware speedup has been measured. All Q/K floating operations, casts, scale,
loop, metadata reads, masks, stores and wrapper guards are retained.

`FPEquiv.block_origin_reuse_equiv` compares the actual original and candidate
grids with an empty experimental rule set. `Proofs/ReuseEquivalence.lean`
proves exact full-state execution equality and actual trace-safety equivalence
at the two changed source assignments, including missing-register failures,
then lifts these facts over every admitted whole-grid permutation. Original
source safety, successful execution and exterior frames remain conclusions.
The main pair keeps `Whole.Valid`, arbitrary strides, aliases, all symbolic
native payloads and empty grids; no finite-value premise is added. Optional
single-source native and Tensor results remain under `Misc/` unchanged.
The concrete Python AST comparison checks the entire wrapper and source after
exactly this assignment-pair rewrite and candidate entry renaming. This is
integer source validation, not a numerical experiment; no experiment is
required by this strict source transformation.

Optional application proofs and readers are under `Misc/`; their declaration namespaces and theorem statements are unchanged. Required main proof helpers remain under `Proofs/`.
