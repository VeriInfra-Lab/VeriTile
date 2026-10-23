# Finite kernel IO and retained Tensor backing

Import `KernelOnly.Correct` for the public finite mathematical kernel IO specifications
and `KernelOnly.Misc.NativeCorrect` for the ordinary native observation theorem.
`KernelOnly.FPEquiv` records the pending program-pair proof; native results require an explicit Misc import.
Import `KernelOnly.Misc.TensorBridge` for the retained shared Tensor backing endpoints:
`forward_tensor`, `output_tensor`, `backward_tensor`, and `gradient_tensor`
in `KernelOnly.TensorBridge.Real` and `KernelOnly.TensorBridge.Native`.
`KernelOnly.Misc.Review` audits both layers; `KernelOnly.Misc.Reader` retains the existing reader checks.

This follow-up closes the two findings in the independent v1 review:
`.veritile-cache/kernel-evidence/chunk_retention_ops/frozen-packet.json` remains unchanged, while
the new endpoints bind all operands and outputs to a common physical backing.
Historical top-level modules and all v1 sources are preserved.

`KernelTensorBacking.Buffers` contains ordinary actual FP32 Tensor views for
exactly the inputs and outputs used by the selected kernel. Optional seed,
terminal, final-state and initial-gradient views are required only when their
source flags are enabled. It requires separate full backing for distinct
outputs and for each input/output pair. Input views can share backing because
they all read the same initial memory.

`snapshot` defines the interpreter input memory directly from those input data
pointers and the one caller `read`. `snapshot_logical` and `snapshot_native`
prove that every in-bounds input word is the corresponding actual Tensor
logical observation, with arbitrary storage offsets retained. There is no
independent numerical input-memory parameter left in the Tensor application endpoints.
Unused regions have a fixed filler; masked source loads and disabled optional
operands retain their existing behavior.

`post` is one physical memory function for the entire executed program. It
updates exactly the union of enabled output write footprints using values from
the same successful quoted state. `written_unique` derives uniqueness from
output backing separation. `post_logical` supplies every output readback;
`Frame` preserves every address outside that union and every word in each
input's full backing, including storage-offset prefixes and tails. All dQ, dK,
and partial-dV observations use this same `post`, as do history and optional
state outputs. Output views cannot give contradictory values by independently
projecting the same address.

`KernelBoundReal` and `KernelBoundNative` reuse the previously checked source
execution and independent mathematical/scalar target proofs. They retain all
casts, FP32 word behavior, local FMA/materialization order, standalone history
inputs, optional outputs, and per-key-block partial dV. No input/output numerical
equation, execution-success premise, or whole compiler-result premise is added.

The fixed primitive/compiler profile remains the native model boundary. This
follow-up adds no host/autograd, allocator, byte-heap, or resource proof, and
does not expand grid-product compatibility. It reuses the saved complete AST
checks and 39 offline installed-binder compilation controls. Its immutable
evidence is `frozen-packet-v2.json` in the task's kernel-evidence directory.

## Public kernel interfaces

`IO.lean` declares eight finite head-local input channels and all eight output
families, actual source selection, exact windows and masks, optional-state
guards, original program and grid coordinates, and an empty private-scratch
list. Source loads, stores and computations remain FP32. The `.real` channel
tag is the existing Structural interpreter representation; the native payload
is a staged FP32 term and its observable value is a 32-bit word.

`Formula.lean` contains the independent dense mathematical and scalar native
formulas. Neither formula calls the source runner or the packet memory result.
`Correct.lean` has four public `Spec.Real (realIO.Implements mathematical)`
statements, one per genuine standalone kernel. Forward-state and output stages
retain arbitrary program IDs. Backward-state retains its original `StatePID`
condition; gradient retains head/chunk bounds and no key-block bound. The
actual launch installs the original B*H grid count. Real and native theorems
cover every history store, optional final/initial state, output, dQ, dK and
per-key-block partial dV; no pipeline or final dV reduction is substituted.

The ordinary `FPEquiv.kernel_native` theorem in `Misc/NativeCorrect.lean` is the
shared `AddressedProgramIO.Implements` native
observation judgment over all four original stages and **arbitrary program
IDs**. This task has an independent source-to-native observation result, not an
optimization pair. Its unchanged `Numeric.ScalarLaws` and `S.LiteralLaws` are
primitive/model boundaries; no floating transformation law or native=Real
claim is introduced. The existing numerical profile and experimental status
are unchanged.

The shared IO judgments include actual source access safety, successful source
execution, typed output values and exact exterior memory frame. The new
`Proofs/IOSafety` follows every actual loop state and load/store.
`Proofs/IOExecution.schedule_result` reuses the existing complete execution
proofs and proves that initial private registers/undefined values are irrelevant.
Finite input binding and masked locality feed those same results into the
independent formulas in `Proofs/IOResults`. The local dV readback proof cancels
the same-program base, so native correctness requires no new head-grid bound.

`../Misc/RawCorrect.lean` and `../Misc/RawNative.lean` preserve all former raw
entry statements, including wider native observation of every memory cell.
`KernelOnly.TensorBridge` retains its exact original Tensor statements and
proofs, with only imports redirected to the supporting helpers. Those
endpoints use the same `KernelForward`, `KernelOutput`, `KernelBackward`,
`KernelGradient` and `KernelPacket` execution foundation as public IO. Their
single physical post-memory, backing separation, offsets, all readbacks and
full exterior/input-backing frames remain unchanged. Public entry import
closures contain no TensorBridge; supporting proofs do not import public
Correct/NativeCorrect readers.

No Python, kernel source, input contract or frozen evidence was changed.
This reorganization adds no numerical/GPU experiment, independent cold replay,
host-runtime project or grid-product compatibility work.

## Program-pair review, 2026-10-21

Program-pair proof connection pending. `FPEquiv.lean` currently supplies no completed two-program
equivalence specification. Historical single-source native results retain
their full theorem names and original statements in `Misc/NativeCorrect.lean`;
import the optional Misc module explicitly for these retained results.

One auxiliary AddressedProgramIO native observation is retained for four stages. Four original benchmark candidates exist, but repaired quotations add signed program/lanes casts, explicit zero padding and reindexed descending loops. No full per-stage source/candidate bridge under the existing arbitrary PID/FP32 word domain is established. Native ScalarLaws/LiteralLaws are model boundaries, not a replacement program or transform proof.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.
