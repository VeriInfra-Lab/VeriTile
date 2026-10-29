# apply_penalty: checked in-place kernel

The original Python benchmark remains unchanged. This revision checks a direct
contiguous-input contract, widens indices before arithmetic, and passes the
original seven tensors to one launch. It neither stages logits/penalties nor
copies results back. Python returns `None`; the original logits allocation is
updated in place.

For each record `k` in row `b`'s cumsum window, with token `t = ids[k]`, the
independent real target is

```
logits[b,t] = (if entry_logits[b,t] > 0 then entry_logits[b,t] / repetition[b]
              else entry_logits[b,t] * repetition[b])
              - counts[k] * frequency[b] - presence[b]
```

Every unselected logit and every byte outside the selected four-byte words is
preserved. This real formula is not an unconditional claim about rounded FP32.

## Checked domain

- Logits are rank-two FP32; the three penalties are rank-one FP32. IDs, counts
  and cumsum are rank-one **int32**, as in the original benchmark's explicit
  dtype declarations. All seven tensors are strided, contiguous, on the same
  CUDA device, and have no unresolved negative view bit. Nonzero storage offsets
  and singleton dimensions with unusual unused strides are allowed.
- Every tensor has fewer than `2**31` elements. Every nonempty data pointer is
  divisible by four. Empty tensors impose no alignment requirement.
- Penalties have at least `B` entries. For `B > 0`, cumsum has at least `B+1`
  entries. Each window obeys `0 <= start <= stop <= min(ids.numel, counts.numel)`.
  Referenced IDs lie in `[0,V)` and referenced counts are nonnegative. Unused
  prefixes/suffixes may contain arbitrary int32 words; capacities may differ.
- IDs are unique **within each row**. Both equal-count and conflicting-count
  duplicates are rejected before launch: equal-value duplicate GPU stores still
  race. Equal token IDs in different rows are allowed.
- Logits' byte interval is disjoint from each read-only argument's interval.
  Read-only arguments may alias one another. Interval checks use actual data
  pointers, so offset views and differently based views are handled.
- `operator.index(hint)` must lie in `[0,65536]`. The block is the original
  512/1024/next-power-of-two choice. Every actual window must fit that block;
  an understated hint is accepted when the minimum/rounded block still covers
  it. The explicit tile cap avoids adding a segmented implementation.
- Zero batches return without launch. Empty rows and zero vocabulary are
  admitted when their metadata windows are empty. Nonfinite logits/penalties
  and zero/negative repetition values remain structurally admitted.

These are explicit restrictions of this revision, not retroactive guarantees
for the original wrapper or a claim to retain the historical copying repair's
broader domain. Contiguity, alignment, resolved views, and byte separation let
the direct kernel implement logical tensor reads without generic copies. The
int32 extent and tile bounds make every widened index representable.

## Kernel proof scope and organization

`Kernels.lean` contains the readable canonical sources with explicit
`tl.region ... = tl.float32` annotations. Both proof entries reference these
sources. Integer channels are the nonnegative projections of the original
signed int32 words. `Proofs/Contract.lean` proves their sign extension, widened
mask correspondence, bounds and exact count-word roundtrip; it does not assume
that conversion of large counts to FP32 is exact.

`RealIO.lean` exposes the two genuine masked gather/scatter IO descriptors.
`Correct.lean` contains the original-source mathematical specification
`apply_penalty_correctness`. `Misc/AuxiliaryCorrect.lean` retains the reciprocal
candidate result `Correct.apply_penalty_reciprocal_correctness` as an ordinary
theorem with its unchanged contract. Both use the independent `lionPenalty`
formula and the shared
`MetaScatterMasked2DKernelIO₁.Implements` judgment. `Proofs/RealIO.lean`
composes the existing complete source execution, actual access safety, active
write injectivity and exterior frame. `Misc/TensorBridge.lean` retains the checked
Tensor trace-safety application and Python admission correspondence.

`IO.lean` exposes the FP32 source and reciprocal candidate as shared
`ProgramIO` schedules. Its channels list three scalar penalties, two cumsum
slots, active IDs/counts and gathered in-place logits; windows and masks depend
only on those logical inputs. `FPEquiv.apply_penalty_reciprocal_equiv` displays
the actual `original stride B ≡[R] reciprocal stride B` transformation and
reuses the existing complete-source theorem in `Proofs/KernelResults.lean`
(`RawFP`). Its program view quantifies every program id and retains the active
positive-arm domain, successful executions, full typed-memory equality and
exterior frame. The ordinary `FPEquiv.apply_penalty_reciprocal_io_equiv` result
retains the shared `ProgramIO.EquivalentOn` application, proving both actual
source traces safe, both executions successful, all active output cells equal
and both exterior frames. `Proofs/PublicEquivalence.lean` reuses the same
existing full-memory equality proof.

Its DIV-MUL-RCP operands are the actual loaded logit and repetition value.
Only lanes with an active store and a true positive comparison need finite
operands/nonzero repetition. Empty windows, inactive padding and unchanged
nonpositive/unordered branches need no such conditions. Repeated token IDs
remain admitted by this floating equivalence endpoint, with exactly the same
store order on both sides; the mathematical endpoint retains its original
active-write injectivity requirement. Backend floating comparison availability
is a semantic capability requirement, not a numerical experiment or input guard.

The existing byte-address connection remains in `Proofs/Memory.lean` and
`Proofs/Binding.lean`: entry words come from the actual data-pointer heap;
logits/metadata byte separation derives read-only stability and the byte frame.
The ordinary valid-allocation law bounds active 64-bit addresses. The four-byte
alignment requirement follows the emitted scalar `.b32` accesses and PTX ISA
8.7 §6.4.1; shared source references are in
[`LOAD_STORE_AUDIT.md`](../add_example/LOAD_STORE_AUDIT.md).

Historical whole-grid/native transport results remain optional in
`Proofs/LegacyGrid.lean`, `Proofs/LegacyNative.lean` and their helpers.
`LocalRounded` and `LocalExact` are optional explicit-premise transport lemmas;
neither is an outstanding task experiment. The public entries do not claim
native arithmetic equals Real or universally correctly rounded IEEE division.

## Numerical assumptions and evidence

The source-level FP32 DIV-MUL-RCP rule already has a published accepted report:
`experiments/floating_point/supplement/report/summary.json#fp32/DIV-MUL-RCP`,
4096 replicates, both gates PASS. Its file SHA256 is
`cea7050c025afc18afa8370ad1e630f3f1d391c5b99dcab44f701adb06708bb6`.
This report is reused under the repository's admission policy; this task did
not independently replay it or establish an unconditional IEEE identity.

The primary kernel results are **complete under the existing admitted rule**.
Their scope is the canonical kernel source. The optional compiler-native
equivalence and its dedicated experiment were removed at the user's request;
no numerical experiment remains pending for this task. This removal does not
establish equivalence of the compiled FMA schedules.

Original benchmark defects remain distinct from this checked revision and
retain their concrete Lean counterexamples: signed-mask overflow, an understated hint,
the demo's too-short cumsum, and conflicting duplicate targets.

## Validation

The accepted targeted builds cover the canonical source, Contract, both public
entries, projection, source execution/memory/frame helpers, counterexamples,
and the retained optional historical modules. Axiom audits use the standard
base only; no `sorry` or new axiom is introduced.

After removing the optional native equivalence on 2026-10-21, targeted builds
of the affected modules and a joint import of the public entries and retained
auxiliary results passed. The joint check also confirms that the removed
theorem and its dedicated namespace are absent from the imported environment.

```
python bench/repairs/tritonbench_g/apply_penalty/check_contract.py
```

The contract probe passes 52 actual-wrapper/kernel-body cases on bounded NumPy
storage, including all tile branches, empty/offset/aliased inputs, complete
frames and rejection before mutation. It checks that the original and revised
production ASTs agree after erasing precisely the int64 index casts, and that
the reciprocal candidate changes only its intended expression. The same probe
parses both canonical Lean DSL bodies and checks their ASTs against the two
Python bodies after removing only region precision declarations and the proved
index-widening casts. Its arithmetic
interpreter covers source FP32 operations, not native FMA validation.

`Correct.lean` and `FPEquiv.lean` are the mathematical and floating reader
entries. Operational derivations and optional historical transport lemmas live
in `Proofs/`; `Misc/TensorBridge.lean` holds the optional Tensor applications. The
existing DIV-MUL-RCP rule and its evidence are unchanged. No new GPU execution
or independent numerical replay is claimed.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/AuxiliaryCorrect.lean](Misc/AuxiliaryCorrect.lean), [Misc/FPObservation.lean](Misc/FPObservation.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.
