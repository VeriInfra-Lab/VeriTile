# FastCE kernel verification

`Correct.lean` covers the three JIT kernels in the unchanged checked `fast_ce_loss_body.py` through independent mathematical `ProgramIO` judgments. `FPEquiv.lean` compares each complete repaired source with its existing integer row-index commutation candidate using `≡[[]]`; every FP32 node, repaired mask and signed-int64 label is retained. `Misc/NativeCorrect.lean` preserves the previous native `ProgramIO` observations as ordinary theorems, including actual access safety, successful execution, outputs and exterior frames. `Contract.lean` retains the Tensor input contract; `Misc/TensorBridge.lean` retains the complete prepared-Tensor applications with actual output reads, native pointer and scalar ABI observations. Wrapper guards, allocation, host chunk aggregation, autograd, and CUDA launch execution remain outside these kernel entrypoints. The former public host files remain in `Proofs/HostContract.lean`, `HostCorrect.lean`, and `HostFPEquiv.lean`.

The transform is `t(x)=a*x` without softcapping, or `cap*tanh(a*x/cap)` with softcapping, where `a` is the nonzero Python scaling value or one. Either sign of nonzero cap/scaling is admitted; either signed zero disables its branch. Full forward writes logical-vocabulary LSE and `LSE-t(target)`, with literal zero loss for ignored labels. Every chunked forward program writes its own active-column LSE; only chunk zero writes `-t(target)` or ignored zero. Backward stores the in-place gradient using the supplied saved LSE and prepared row upstream Tensor. The final ignored-label selection is outside multiplication and covers all FP32 words, including NaNs and infinities.

Inputs retain FP32 words, signed-int64 labels, actual Tensor shape/strides/storage offset, allocation bytes, active data-pointer alignment, and device identity. No tensor-finiteness guard or backing-base alignment assumption is added. Actual unused singleton strides are retained; `flat_vector`, `flat_matrix`, and native address proofs establish active indexing. Partial allocation byte tails and nonzero storage offsets are admitted. Empty row grids are vacuous over launched program IDs; the checked wrapper's explicit positive denominator still admits empty/all-ignored inputs. External upstream may alias logits; the kernel boundary receives the actual prepared row upstream allocation, including stride-zero expansion.

Single forward uses `nextPowerOfTwo(vocab)` and its selected 4/8/16/32 warps. Chunked forward uses width 65536 and 32 warps. Backward uses width 4096 and 8 warps. The checked bounds are rows at most `2^31-1` and `ceil(vocab/4096)` at most 65535.

The ideal-real layer is separate from the native word layer. Native logical channels admit arbitrary symbolic terms, including actual input word leaves; the existing model retains libdevice tanh, exp/log, source operand order, and complete input-covering reduction trees. Task-local lowering preserves the source cast graph, removes same-FP32 casts before selection, and handles both `c-x*y` and `x*y-c` FMA orientations. Tagged Python int64 and binary64 payloads remain explicit; only per-instruction conversion obligations connect them to typed source literals. Branch selection uses Python nonzero before FP32 rounding. No FMA/split equivalence, reassociation, FP-to-real accuracy, or GPU numerical conformance is asserted.

Validation includes scoped Lean build receipts, a public-only `Misc/KernelAudit.lean`, ten SM80 Triton compile probes with pinned TTIR/TTGIR/LLVM/PTX artifacts, the unchanged CPU suite (47 rejected-input checks and 25 autograd/source/frame cases), and six direct-kernel CPU probes. The CPU interpreter uses a NumPy tanh port. Direct probes cover V65537 chunk contributions, supplied backward LSE/upstream, exceptional ignored zeros, signed-zero options, offsets/frames, and unused strides of `2^62`.

The original padded-softcap and missing chunk-LSE-store counterexamples are retained in the original task's `Proofs/Counterexamples.lean`. The CPU suite also reproduces the earlier ignored-NaN gradient defect (`0*NaN`), fixed in the selected body by the explicit final zero selection. Python body, checked wrapper, and guard tests are unchanged.

## Kernel and Tensor interfaces

`Correct.lean` declares visible `forwardIO`, `chunkedIO` and `backwardIO` records,
independent LSE/loss/gradient formulas, and one `specification` for each actual
source kernel. The shared `ProgramIO.ImplementsOn`/`Implements` conclusions
include source-access safety, successful execution, all active typed outputs,
and the exact exterior cell frame. These retain the previous per-program
scope, including every source flag combination and tail; they do not replace
a previous full-grid result.

`Misc/NativeCorrect.lean` uses `ProgramIO.ObservesOn`/`Observes` for the existing native
observation of the source. The logical carrier is
still arbitrary `Term`, and `R`, `I`, `S` retain the existing reduction,
instruction and compiler model boundaries. The native forward/chunked label
domain remains `label = -100 ∨ 0 ≤ label`: the separately backed target-gather
channel deliberately adds no vocabulary upper bound. The Real label domain
retains its existing vocabulary range. Ignored labels suppress the gather and
upstream reads; nonfirst chunks suppress loss writes and target reads. Inputs
and in-place output aliases remain explicit in the IO records.

The complete prepared-Tensor results remain byte-for-byte unchanged in
`TensorBridge.Real.forward`, `chunked`, `backward`, and their native counterparts.
All Tensor/label-byte/native-address/scalar-ABI/output/frame observations remain.
The public IO proofs and Tensor bridge reuse the same `KernelExecution` and
`KernelMath` foundations. Raw interfaces stay in `Proofs/KernelInterface.lean`;
scalar FMA and ignored-zero derivations stay in `Proofs/KernelFMA.lean` and
`KernelOutputs.lean`. Neither main entry imports the Tensor bridge.

`Misc/Audit.lean` is the public reader audit. `Misc/KernelAudit.lean` checks the
preserved foundations without importing public entries. The format-only work
uses targeted Lean builds and formula/axiom/import audits; it adds no numerical
law, experiment, independent replay or GPU run. The earlier validation results
listed above retain their original historical scope.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [NativeCorrect.lean](Misc/NativeCorrect.lean), [Audit.lean](Misc/Audit.lean), [KernelAudit.lean](Misc/KernelAudit.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.

## Original Python function bodies

`Kernels.lean` contains six complete pasted Python bodies: the three selected
JITs in `fast_ce_loss_body.py` and their existing row-index commutation
candidates in `candidate_kernels.py`. Each body has all 17 top-level statements.
Original casts, names, one-line constexpr branches and statement order remain.
Caller pointer/scalar types, constexpr bindings and the `triton_tanh` import
binding live outside `triton { ... }`. The candidate functions retain the
original signatures and both the `triton.heuristics` and `triton.jit` decorators;
only the established integer row-offset product is commuted.

The selected wrapper loads `fast_ce_loss_body.py` with a literal local
`importlib` loader. The corpus checker statically resolves and pins both files
and requires correspondence for every JIT in the loaded file. It never executes
the loader. Import rebinding, nonliteral file selection and incomplete body
coverage cannot certify correspondence. The separate libdevice alias is checked
against its actual Python import and assignment.

The previous Lean source specialized away the two constexpr conditionals and
added explicit terminal `.to(tl.float32)` conversions absent from Python. The new source
keeps both conditionals and the original plain stores. Internal execution
expressions now describe those actual stores. The backward store retains its
existing implicit conversion of the computed `tl.where` expression to the
pointer dtype in the execution model; this adds no text to the pasted body.
No arbitrary floating-point cast
identity is assumed. Independent real formulas and public correctness,
equivalence, IO and contract declarations retain their prior source. Internal
proofs are replayed against the complete new source bodies.

The new frontend regression compares one-statement inline suites with their
indented counterparts and rejects unsupported multi-statement inline suites.
The Python regression mutates casts, constexpr guards, caller bindings, import
aliases and candidate decorators to check that source drift is rejected.
Current migration verification is recorded separately from the historical
numerical and compiler experiments above; no new GPU experiment is claimed.
