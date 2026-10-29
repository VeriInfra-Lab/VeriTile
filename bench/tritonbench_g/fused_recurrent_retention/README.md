# fused_recurrent_retention

The public [Correct](Correct.lean) and [FPEquiv](FPEquiv.lean) entries now expose
the complete forward/backward source contracts. Their
[isolated public rebuild](../../review-checkpoints/2026-10-13/retention-public-isfinite-candidate-stage/receipt.json)
passes, followed by the [4,899-job root aggregate](../../review-checkpoints/2026-10-14/public-barrier-aggregate-root/receipt.json)
with the integrated barrier semantics.
Historical single-step slices and prepared-carry results are preserved in
[Proofs/LegacyReal.lean](Proofs/LegacyReal.lean), separate from these public claims.

The new [`Kernels.lean`](Kernels.lean) keeps both complete original
kernel bodies, all flags, masked loads, storage casts, pointer updates, the
backward barrier and optional final-state store. Each candidate reverses only
the operands of the discrete key-mask comparison. The source macros expose
independent storage parameters; the current named constructors use a common
input type and an independent output type.

Source, parameter model, exact phase decomposition, independent masked
initial-memory formulas and active-lane address uniqueness have passed local
Lean checks. The formulas include both DK/DV tails and seeded state.

The [complete forward real checkpoint](../../review-checkpoints/2026-10-13/fused-retention-forward-real/receipt.json)
proves actual source initialization, all 11 loop statements and optional final
state writeback for the erased fp32 profile. Its independent initial-memory
formulas cover seeded/unseeded state, feature tails, all T including zero,
and a complete tagged-cell frame.

The [finite-memory checkpoint](../../review-checkpoints/2026-10-13/fused-retention-forward-io/receipt.json)
connects that execution to a flat allocation with actual masked-access safety,
active-address bounds and allocation disjointness. The source's explicit
other=0 loads allow internal oracle normalization without a public assumption
on initial undef values. Input/output and output/final separation hypotheses
remain explicit.

The [backward finite-memory checkpoint](../../review-checkpoints/2026-10-13/retention-backward-finite-io/receipt.json)
proves both complete loops, the barrier and pointer rebases, successful flat
execution, independent initial-memory formulas for `dq/dk/dv` and the full
tagged-cell frame. All six final pointer retreats execute unchanged. The bridge
preserves strict state correspondence while another iteration remains and
uses memory/launch observation equality after the final dead-pointer updates.
The five new leaves first passed private local compilation with 193 unchanged
inputs. The [integrated isolated rebuild](../../review-checkpoints/2026-10-13/retention-typed-isfinite-parametric-stage/receipt.json)
then passed ordinary Lake checking with the exact debug-barrier semantics.
The public migration and root aggregate replay both pass.

The [independent typed equivalence checkpoint](../../review-checkpoints/2026-10-13/fused-retention-state-equivalence/receipt.json)
passes five axiom gates for complete forward/backward original-candidate
state equality under arbitrary structural algebra and initial state. It
does not assume real arithmetic identities.

The [typed forward checkpoint](../../review-checkpoints/2026-10-13/fused-retention-typed-forward/receipt.json)
also proves successful complete original/candidate execution for arbitrary
structural algebra, initial state, tile sizes, T and both flags in the selected
fp32 storage profile. It constructs the actual source-produced carry. A
mathematical backend witness starts with empty registers, nonzero program IDs
and both feature tails. Its casts are identities; IEEE/native refinement,
other-storage successful witnesses and whole-task acceptance remain open.
The same isolated rebuild proves complete typed fp32 backward success for
arbitrary initial states, including both loops and the original barrier.
The precise `tl.debug_barrier` marker preserves sequential per-program state;
unknown and asynchronous effect markers still fail. This does not prove native
thread convergence or inter-program synchronization. The barrier change is
validated in isolation and in the completed root aggregate replay.

The original [`fused_recurrent_retention.py`](fused_recurrent_retention.py)
matches `thunlp/TritonBench@603e28a` byte for byte, with SHA-256
`369709736fce34c980e29fbf8db9b2117eaa5b22d65acad20ea3a898956426f5`.
The full source AST check passes both kernels at all 36 common-input/independent-output
storage combinations each, and rejects 792 mutated negative controls. This
establishes source agreement for those profiles, not native compiler refinement.

The [offline native matrix](../../review-checkpoints/2026-10-13/fused-retention-native/receipt.json)
records 432 compiled and 432 rejected unchanged-source configurations.
The tested fp16/bf16/fp32/fp64 inputs compile at valid wrapper block shapes;
fp8 inputs fail the original masked zero conversion, and the original
non-power-of-two 3/5 block dimensions are rejected. Mixed output dtypes are
direct-JIT profiles, since the wrapper allocates outputs with q's dtype.
This is compilation evidence; no kernels were launched.

The source addresses rows by literal DK/DV increments; its four time/channel
stride arguments are unused. Host layout, cross-program reductions, runtime
scalar ABI, fixed-width overflow and IEEE/native numerical behavior remain
separate obligations. The Python backward wrapper ignores `d_final_state` and
returns no initial-state gradient, so a proof of its actual kernel outputs
must not be described as an autograd proof for arbitrary final-state losses.
