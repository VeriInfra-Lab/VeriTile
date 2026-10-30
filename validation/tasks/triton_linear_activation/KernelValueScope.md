# Kernel numerical scope

The new public kernel theorems use the selected checked Python source and
actual prepared pointers. Default FP32 and explicitly cast FP16 are distinct
storage modes. Caller AMP preparation and saved-buffer copyback are existing
host behavior, not new proof scope.

## Derived results

The complete readable source executes successfully for every admitted valid
program. Every active output and optional saved cell has the independent
reference value. Every other tagged memory cell is unchanged. Canonical
strides derive active store injectivity; masked lanes need not have distinct
formed offsets. K=0, K tails, arbitrary activation strings/None, and all 98
source configurations are covered.

The independent Real target is a dense mathematical sum over K and the named
activation. The source arithmetic proof does not reuse the historical fused
floating graph. The native target is an independently indexed actual-Tensor
expression recurrence, evaluated from exact input bytes.

## Native semantics

`KernelNativeModel.Term` records source precision, typed casts, literal tags,
ordered scalar operations, and two-argument dot operands. An explicit
`CompilerPolicy` evaluates that graph. Dot/add contraction seeds the instruction
sequence with the previous accumulator; source-order evaluation seeds dot with
zero and performs the later add. Scalar multiply/add contraction calls concrete
FP32 FMA. These are different evaluators; no floating identity between them is
assumed or proved.

`instructionDot` constructs arbitrary block K from a chronological sequence
of named fixed instruction lanes. `instruction_coverage` proves exact
K8 TF32/K16 FP16 divisibility for all source configurations. TF32 consumes raw
FP32 operands; no invented TF32 conversion is inserted. FP16 instructions
consume actual low-16-bit half storage words. `KernelBits.wordAt` fixes the
input little-endian byte codec and proves the half container's width.

Concrete scalar definitions cover binary32 literals, FP32 add/sub/mul/FMA,
maxnum, half pack/unpack, and all source storage casts. The only external
numeric functions are local `div.full.f32`, libdevice tanh/erf, and one result
lane of a named fixed K8/K16 ISA instruction. They are explicit primitive
semantics of the model. There is no whole-dot, matrix-result, successful-run,
activation-vector, or completed-output equality assumption.

The two sqrt expressions in the DSL are tags for pinned host constants, whose
native FP32 bits are checked: sqrt(2) = 0x3fb504f3 and sqrt(2/pi) = 0x3f4c422a.
The audit also checks zero, one, half, and cubic coefficient literal bits.

## Ordinary boundaries

Shared actual FP32 Tensor laws, local FP16 Tensor API laws, prepared allocation
observations, tagged-memory/byte input bindings, and compiler/primitive semantics
are explicit model boundaries. No GPU run or compiler implementation
conformance is claimed. Frames do not assert a general post-byte-memory alias
coherence theorem. Proofs are for one invocation at an arbitrary valid program;
a separate grouped-coverage theorem covers all logical output coordinates.

The Real projection is input-only. Finite native values project by decoding;
exceptional values project to zero. The Real theorem proves ideal execution
against the independent mathematical target on that projection; it does not
assert native/Real value equality or an approximation bound.

There are no new numerical equality assumptions or numerical experiment TODOs
in the public kernel statements. The older `FPValueScope.md` and
`numerical-obligations.json` are historical host/compiler-model records and
are preserved unchanged.
