# Scalar ABI bound attention revision

Current selected public entries: [Contract.lean](../../../bench/tritonbench_g/triton_attention/Contract.lean), [Correct.lean](../../../bench/tritonbench_g/triton_attention/Correct.lean), [FPEquivBound.lean](../../../bench/tritonbench_g/triton_attention/FPEquiv.lean). The [contract review](../../contracts.json) records their current scope and assumptions.

## Historical implementation and audit notes

The notes below describe earlier revisions and their recorded boundaries. Their old file lists, receipts and completion counts are historical; they do not establish the current selected contract.

The current mathematical specification and native auxiliary use finite shared IO.
The frozen v1/v2 evidence packets remain historical records. The selected
public certificates are:

- `Correct.kernel_real` and `TensorBridge.tensors_real`.
- `FPEquivBound.kernel_native` and `TensorBridge.tensors_native`.

The original `FPEquiv.kernel_native` and `TensorBridge.Unbound.tensors_native`
are generic internal structural lemmas reused by v2. They are not the v2 actual-scalar-ABI certificates.

Both bound native endpoints require the task-local
`Native.ScalarBinding scalar literal`. Its two facts connect only the input:
the staged source scale leaf must evaluate to `scalar.word`, and raw Python
integer `1` must project to literal `1`. The existing `scalar.marshalled` field
then identifies the same word with `scalar.marshal scalar.raw`. Fixed zero/one
words are included in the binding, so an arbitrary literal callback cannot
bypass those branches. The native target's scale leaf is bound to that word
as well. These are ordinary scalar conversion/specialization laws, not
whole-graph, source-execution or output equalities.

Each public native result now also derives `SourceScale`: every forward or
backward source owner reads the bound scale word from the shared primitive
literal port. Local theorems establish constexpr-one multiply elimination and
backward subtraction, runtime-float-one multiplication and backward FMA, and
preservation of loaded-word provenance. The v1 review's raw-one/project-two
witness cannot satisfy the new interface. `one_binding` constructs the law
for consistent one observations for any unrelated literal conversion.

The original v2 evidence reused the complete three-kernel execution and
independent Real/native targets. The current public IO pass also preserves the
actual Tensor storage readback, object identities and frames.
The Python guard, JIT bodies and existing 9-admit/11-reject scalar controls and
four offline compiler profiles remain unchanged. No new numerical experiment
or GPU execution is needed or claimed.

The local scalar law is explicit in the public theorem assumptions. Actual
Python conversion, compiler profile and instruction meanings, ordinary Torch
API laws and runtime conformance remain the existing external boundaries.
No host-wrapper implementation proof, universal compiler proof, byte-heap
model or new numerical obligation is introduced. The selected proof domain
remains the FP32 wrapper branch with its existing FP16 internal casts.

The additive frozen manifest is
`.veritile-cache/kernel-evidence/triton_attention_scalar_abi_checked-v2/delivery.json`.
It binds the unchanged v1 packet and every added file and successful build
receipt. Root retains integration, acceptance and ledger ownership.

The bound native auxiliary `FPEquivBound.kernel_native` and the generic
`FPEquiv.kernel_native` now live as ordinary theorems in `Misc/NativeCorrect.lean`,
with their original full names and complete statements. Both former reader
files do not import the optional native entry. Their actual Tensor applications remain in optional
`Misc/TensorBridge.lean` as an ordinary theorem with the identical `ScalarBinding`
premise. The Real Tensor application also moves there. `Misc/ReviewEntryV2.lean`
audits these current names; neither kernel certificate imports the bridge. The
frozen V1/V2 packets and historical reviews are unchanged.

## Current public IO organization

`IO.lean` declares Q/K/V/DO plus initialized-zero DQ, four returned arrays
Out/DQ/DK/DV, and L/M/NewDO/Delta scratch. All are finite FP32 windows; the
internal FP16 casts stay in the unchanged source. DQ is an explicit input/output
alias. Every original complete legal forward, preprocess and backward schedule
is retained. No source dtype, Python guard or scalar ABI was changed.

`Spec.lean` gives independent causal-attention/VJP and ordered native formulas.
`Correct.lean` contains the mathematical kernel specification and
`Misc/NativeCorrect.lean` retains both ordinary native theorems using shared
`MaterializedProgramIO.ImplementsOnCells` for the bound observation. It retains
`ScalarBinding` and derives `SourceScale`; it is a source-to-native observation,
not a new optimization or an assertion that floating arithmetic equals Real.
The shared materialized invocation executes each original quotation and lowers
its stored expression at the original scalar owner before the next CTA.

`Proofs/IOSafety.lean` connects the declared bounds to actual forward,
preprocess and both backward loops. `Proofs/FiniteExecution.lean` transports the
prior complete execution results to finite inputs: a proof-only extension of
input memory beyond the tensor does not affect any finite source result.
This is proved by task-local recurrence agreement; no additional input or
numerical assumption is introduced. The exact original exterior frame remains.

`Misc/TensorBridge.lean` retains all prior actual Tensor statements and explicitly
connects their finite input snapshot and complete execution to the shared IO
schedule. `Proofs/LegacyCertificates.lean` retains the previous operational
signatures as ordinary theorems. Helpers import no public reader entry.
`Misc/ReviewEntryV2.lean` checks the current selected entries and Tensor applications.
No new numerical replay, GPU experiment or compiler claim accompanies this
format pass.

Current two-program review: No alternate transformed forward/preprocess/backward pipeline has been specified for a two-program comparison. The single-source native certificates remain useful ordinary theorems with their original names, domains, precision, schedules, and proofs. This missing comparison target or connection is not a numerical-experiment deferral. No Python source, numerical rule, or input contract changed.

Optional proof entries are in [Misc/TensorBridge.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/d67b1d0444c2e0fb8066414c2041129b53bf2931/bench/repairs/tritonbench_g/triton_attention_scalar_abi_checked/Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/d67b1d0444c2e0fb8066414c2041129b53bf2931/bench/repairs/tritonbench_g/triton_attention_scalar_abi_checked/Misc/NativeCorrect.lean), [Misc/ReviewEntryV2.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/d67b1d0444c2e0fb8066414c2041129b53bf2931/bench/repairs/tritonbench_g/triton_attention_scalar_abi_checked/Misc/ReviewEntryV2.lean), [Misc/ReviewEntry.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/d67b1d0444c2e0fb8066414c2041129b53bf2931/bench/repairs/tritonbench_g/triton_attention_scalar_abi_checked/Misc/ReviewEntry.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.
