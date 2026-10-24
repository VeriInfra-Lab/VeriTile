# masked_select

Read `Kernels.lean`, `Correct.lean` and `FPEquiv.lean` first. The selected
production scope is FP32 payloads, Bool masks and the Int64 metadata produced by
Bool `cumsum`. The source Python file is unchanged.

The independent target broadcasts the two logical tensors, flattens them in
row-major order, filters the indices whose mask is true, and gathers their
input values in that order. `stableResult` is defined by list filter/map and
has a checked non-circularity audit. The output length is the number of true
mask elements. The raw FP32 theorem preserves all 32-bit payload patterns,
including signaling NaNs and signed zero.

| Entry | Public result |
| --- | --- |
| `Correct.lean` / `masked_select_real` | Mathematical stable-selection result of the complete host/grid composition, using the shared source's real projection. |
| `FPEquiv.lean` / `masked_select_fp32` | The complete returned FP32 tensor, exact raw payloads, exterior/pre-existing-memory frames, and finite source index/access correspondence. |
| `Kernels.lean` | Source-only original and typed seven-statement kernels, source configurations and grid count. Host interpretation is in `Proofs/HostSource.lean`. |

The host model threads broadcast/contiguous/ravel values into an **inclusive**
Bool-to-Int64 cumulative buffer. The output length is read from that actual
produced buffer at `n-1`. It executes every trial's actual typed source and then
the selected grid, threading returned memory throughout. Trials may choose any
of the original 20 width/warp configurations. Their output is not restored:
input/mask/prefix preservation and a final complete grid prove the same stable
result for every permitted schedule and trial sequence.

The public finite domain is `0 < broadcasted_numel < 2^31`, same-device tensors,
valid source configurations and complete grid permutations. Every padded lane
has exact signed 32-bit index/mask behavior, including inactive lanes. The
prefix and selected destination are exact Int64 values. Contiguous allocation
contracts fix byte widths FP32/Bool/Int64/output to `4/1/8/4`, required alignment,
64-bit address bounds and fresh output. Read allocations may alias each other;
the kernel never writes them.

Torch logical view/broadcast/contiguous/ravel/cumsum semantics, successful
ordinary allocation/device operations, and compiler/runtime/load/store contracts
are the trusted primitive boundary. `prepare` expresses those primitive effects;
the task-specific producer/consumer wiring and actual kernel/grid result are
proved. Owned scratch is restricted to the three actual producer windows and
is fresh relative to pre-existing allocations. A reused contiguous buffer must
already contain the same values. Both public results preserve every cell outside
owned scratch and output `[0:length)`, and every pre-existing allocated cell.
No final-output or successful-kernel callback is assumed.

Failure cases remain distinct. Incompatible shapes fail at the original host
assertion. A broadcasted input with zero elements fails at `prefix_sum[-1]`,
after the producer stages and before output allocation/launch; no unchanged
host-memory claim is made for this failure. A nonempty all-false mask succeeds
with length zero. A zero cumulative prefix produces native destination `-1`,
but that lane is unselected and performs no store. Non-Bool masks are outside
this semantic API; the Python source does not dynamically assert their dtype.
Resource/device failures belong to the ordinary primitive boundary; the host
execution theorem uses its successful-primitive interpretation.

`Proofs/StableSelection.lean` proves the independent filter/count properties.
`Proofs/WholeExecution.lean` derives destination injectivity from the producer's
Bool cumulative values and composes the actual per-block interpreter.
`Proofs/NativeDomain.lean` handles finite indices, signed destinations and byte
windows. `Proofs/HostComposition.lean` connects producer ownership, variable-width
trials, final grid, last-prefix scalar and returned tensor.

All seven added modules have targeted Lean builds and complete module axiom
audits. The public claims depend only on the standard Lean base axioms; there
are no numerical assumptions to defer. The FP-rewrite-specific
`#print_fp_assumptions` command does not accept a structural IO theorem, so the
complete structural dependency audit is the relevant check here. Compilation
does not claim GPU execution or independent native compiler verification.

An independent source replay identified a Triton 3.7.1 **CPU interpreter**
limitation: vector Python `and` is evaluated through a scalar truth conversion,
whereas the native frontend lowers it to elementwise logical conjunction. Any
CPU replay adapting that BoolOp must identify the adaptation explicitly. This
does not change the Python source or the conjunction in the shared proof source.

Legacy `MaskedSelect.lean`, `FPCorrect.lean`, `RawBitCorrect.lean` and their helpers
remain available for compatibility and their original, often wider, consumer
scopes. They are reused rather than rebuilt or removed in this change. Their
top-level placement remains organization debt; they do not create additional
mandatory payload dtype variants or extra task identities.

## Current integration and checked revision

The accepted whole-task helpers and public entries were migrated into this
workspace without re-proving their established results. Source definitions were
separated into `Kernels.lean`; their names and bodies remain unchanged. The
original Python is unchanged. Targeted compilation of the migrated dependency
chain and both public entries passed.

The historical original theorem remains conditional on its stated producer and
finite-domain contracts. It does not retroactively add Python input guards.
In particular, actual `.contiguous()` can retain a lazy-negative bit, so the
logical producer contract alone does not certify that source case. The separate
[checked revision](../../repairs/tritonbench_g/masked_select/README.md) resolves
that case and exposes the finite-domain and prepared-alignment checks. No
historical acceptance ledger or numerical status is changed by this integration.
