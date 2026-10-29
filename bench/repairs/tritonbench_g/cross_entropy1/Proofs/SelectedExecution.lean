import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Lowering
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NegativeStorage

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Identity of one actual ordinary API call. Full tensor descriptors, not
just reduction length, participate in the platform's selection. -/
structure CallDescriptor (σ : Type) where
  invocation : Nat
  before : σ
  after : σ
  inputs : List Tensor
  outputs : List Tensor
  implementation : String
  objectDigest : String
  device : Nat
  stream : Nat
  rank : Nat
  group : Nat
  work : Option Nat
  configuration : List Nat

/-- An occurrence is a particular output coordinate / primitive within a
selected call. Repeated rows, calls, devices and ranks need not share a tree. -/
structure Occurrence (σ : Type) where
  call : CallDescriptor σ
  coordinate : List Nat
  instruction : Nat

inductive CollectiveAlgorithm where
  | ring | tree | collNetDirect | collNetChain | nvls | nvlsTree | network
  deriving DecidableEq

inductive ReductionKind where
  | laneSum | laneMaximum | torchSum | torchMaximum | torchExponentialSum
  | collective (algorithm : CollectiveAlgorithm)
  deriving DecidableEq

def ReductionTree.leaves : ReductionTree width → List (Fin width)
  | .input i => [i]
  | .identity _ => []
  | .combine a b => a.leaves ++ b.leaves
  | .transfer _ a => a.leaves

def ReductionTree.padding (identity : Word) : ReductionTree width → Prop
  | .input _ => True
  | .identity x => x=identity
  | .combine a b => a.padding identity ∧ b.padding identity
  | .transfer _ a => a.padding identity

/-- Legal schedule data: each actual input occurs once; padding has the
specified identity bits. This is purely structural, with no arithmetic law. -/
structure LegalTree (identity : Word) (tree : ReductionTree width) : Prop where
  inputs : tree.leaves.Perm (List.finRange width)
  padding : tree.padding identity

/-- The platform chooses these descriptors/trees for this particular call.
Neither this selection relation nor its descriptors contain numerical outputs.
For NVLS/network profiles the tree is the selected local arithmetic behavior;
its scalar numerical obligations are identical to other binary rank adds. -/
structure Selection (σ : Type) where
  invoked : CallDescriptor σ → Prop
  reduction : (site : Occurrence σ) → ReductionKind → (width : Nat) → ReductionTree width → Prop

structure SelectedTree {σ : Type} (platform : Selection σ) (kind : ReductionKind)
    (identity : Word) (width : Nat) where
  site : Occurrence σ
  tree : ReductionTree width
  called : platform.invoked site.call
  selected : platform.reduction site kind width tree
  legal : LegalTree identity tree

/-- Primitive trace identities. The trace includes the actual instruction or
local hardware reduction step and its operand events. Reads and stores retain
native state, exact object descriptor and logical coordinate. -/
structure PrimitiveTrace (σ : Type) where
  bits : Nat → Word
  raw : Occurrence σ → Word → Nat → Prop
  load : Occurrence σ → σ → Tensor → Nat → Nat → Prop
  literal : Occurrence σ → ℝ → Nat → Prop
  natural : Occurrence σ → Nat → Nat → Prop
  negativeInfinity : Occurrence σ → Nat → Prop
  binary : Occurrence σ → ScalarBinary → Nat → Nat → Nat → Prop
  unary : Occurrence σ → ScalarUnary → Nat → Nat → Prop
  fma : Occurrence σ → Nat → Nat → Nat → Nat → Prop
  hostScale : Occurrence σ → ℝ → Nat → Nat → Prop
  transfer : Occurrence σ → Transfer → Nat → Nat → Prop
  store : Occurrence σ → Nat → σ → Tensor → Nat → Prop

/-- Ordinary native interface for one fixed-arity step. In particular there
is no list/graph/reduction evaluator among these fields. -/
structure PrimitiveContracts {σ : Type} (trace : PrimitiveTrace σ) (native : Numerics)
    (readWord : σ → Tensor → Nat → Word)
    (readByte : σ → Tensor → Nat → Fin 4 → Physical.Byte) : Prop where
  raw : ∀ s x out, trace.raw s x out → trace.bits out=x
  load : ∀ s state tensor index out, trace.load s state tensor index out →
    trace.bits out=readWord state tensor index
  literal : ∀ s x out, trace.literal s x out → trace.bits out=native.literal x
  natural : ∀ s x out, trace.natural s x out → trace.bits out=native.natural x
  negativeInfinity : ∀ s out, trace.negativeInfinity s out → trace.bits out=native.negativeInfinity
  binary : ∀ s op a b out, trace.binary s op a b out →
    trace.bits out=op.run native (trace.bits a) (trace.bits b)
  unary : ∀ s op a out, trace.unary s op a out → trace.bits out=op.run native (trace.bits a)
  fma : ∀ s a b c out, trace.fma s a b c out →
    trace.bits out=native.fma (trace.bits a) (trace.bits b) (trace.bits c)
  hostScale : ∀ s scale a out, trace.hostScale s scale a out →
    trace.bits out=native.hostScale scale (trace.bits a)
  transfer : ∀ s edge a out, trace.transfer s edge a out → trace.bits out=trace.bits a
  store : ∀ s value state tensor index, trace.store s value state tensor index → ∀ byte,
    readByte state tensor index byte=Physical.byteOf
      (if tensor.isNeg then Physical.flipSign (trace.bits value) else trace.bits value) byte

/-- An execution is built only from primitive events. Inlining an already
stored subgraph may reuse its earlier event id, so load barriers and repeated
uses retain the original selected reduction occurrence. -/
inductive Execution {σ : Type} (trace : PrimitiveTrace σ) (readWord : σ → Tensor → Nat → Word) (inputs : Fin width → Word) :
    ScalarGraph width → Nat → Prop
  | input (i : Fin width) (site state tensor index out)
      (event : trace.load site state tensor index out)
      (binding : readWord state tensor index=inputs i) : Execution trace readWord inputs (.input i) out
  | loaded (site state tensor index x out)
      (event : trace.load site state tensor index out)
      (binding : readWord state tensor index=x) : Execution trace readWord inputs (.raw x) out
  | raw (site x out) (event : trace.raw site x out) : Execution trace readWord inputs (.raw x) out
  | literal (site x out) (event : trace.literal site x out) : Execution trace readWord inputs (.literal x) out
  | natural (site x out) (event : trace.natural site x out) : Execution trace readWord inputs (.natural x) out
  | negativeInfinity (site out) (event : trace.negativeInfinity site out) :
      Execution trace readWord inputs .negativeInfinity out
  | binary (site op a b av bv out) (left : Execution trace readWord inputs a av)
      (right : Execution trace readWord inputs b bv) (event : trace.binary site op av bv out) :
      Execution trace readWord inputs (.binary op a b) out
  | unary (site op a av out) (value : Execution trace readWord inputs a av)
      (event : trace.unary site op av out) : Execution trace readWord inputs (.unary op a) out
  | fma (site a b c av bv cv out) (left : Execution trace readWord inputs a av)
      (right : Execution trace readWord inputs b bv) (addend : Execution trace readWord inputs c cv)
      (event : trace.fma site av bv cv out) : Execution trace readWord inputs (.fma a b c) out
  | hostScale (site scale a av out) (value : Execution trace readWord inputs a av)
      (event : trace.hostScale site scale av out) : Execution trace readWord inputs (.hostScale scale a) out
  | transfer (site edge a av out) (value : Execution trace readWord inputs a av)
      (event : trace.transfer site edge av out) : Execution trace readWord inputs (.transfer edge a) out

theorem Execution.native_value {σ : Type} {trace : PrimitiveTrace σ} {native : Numerics}
    {readWord readByte} (api : PrimitiveContracts trace native readWord readByte)
    {inputs : Fin width → Word} {graph : ScalarGraph width} {out : Nat}
    (execution : Execution trace readWord inputs graph out) : trace.bits out=graph.eval native inputs := by
  induction execution with
  | input i site state tensor index out event binding => exact (api.load site state tensor index out event).trans binding
  | loaded site state tensor index x out event binding => exact (api.load site state tensor index out event).trans binding
  | raw site x out event => exact api.raw site x out event
  | literal site x out event => exact api.literal site x out event
  | natural site x out event => exact api.natural site x out event
  | negativeInfinity site out event => exact api.negativeInfinity site out event
  | binary site op a b av bv out _ _ event ihA ihB =>
    simpa only [ScalarGraph.eval,ihA,ihB] using api.binary site op av bv out event
  | unary site op a av out _ event ih =>
    simpa only [ScalarGraph.eval,ih] using api.unary site op av out event
  | fma site a b c av bv cv out _ _ _ event ihA ihB ihC =>
    simpa only [ScalarGraph.eval,ihA,ihB,ihC] using api.fma site av bv cv out event
  | hostScale site scale a av out _ event ih =>
    simpa only [ScalarGraph.eval,ih] using api.hostScale site scale av out event
  | transfer site edge a av out _ event ih =>
    simpa only [ScalarGraph.eval,ih] using api.transfer site edge av out event

theorem Execution.reference_value {σ : Type} {trace : PrimitiveTrace σ} {native reference : Numerics}
    {readWord readByte} (api : PrimitiveContracts trace native readWord readByte)
    (ports : Obligations native reference) {inputs : Fin width → Word} {graph : ScalarGraph width} {out : Nat}
    (execution : Execution trace readWord inputs graph out) : trace.bits out=graph.eval reference inputs :=
  (execution.native_value api).trans (graph.equiv ports inputs inputs (fun _ => rfl))

/-- A local store, followed by byte-wise lifetime/frame preservation to the
observed returned object. No expected whole-output value occurs in the API. -/
structure StoredExecution {σ : Type} (trace : PrimitiveTrace σ)
    (readWord : σ → Tensor → Nat → Word) (readByte : σ → Tensor → Nat → Fin 4 → Physical.Byte)
    (inputs : Fin width → Word) (graph : ScalarGraph width)
    (state : σ) (tensor : Tensor) (index : Nat) where
  output : Nat
  execution : Execution trace readWord inputs graph output
  site : Occurrence σ
  stored : σ
  store : trace.store site output stored tensor index
  retained : ∀ byte, readByte state tensor index byte=readByte stored tensor index byte

theorem StoredExecution.reference_bytes {σ : Type} {trace : PrimitiveTrace σ} {native reference : Numerics}
    {readWord readByte} (api : PrimitiveContracts trace native readWord readByte)
    (ports : Obligations native reference) {inputs : Fin width → Word} {graph : ScalarGraph width}
    {state : σ} {tensor : Tensor} {index : Nat}
    (stored : StoredExecution trace readWord readByte inputs graph state tensor index) (byte : Fin 4) :
    readByte state tensor index byte=Physical.byteOf
      (if tensor.isNeg then Physical.flipSign (graph.eval reference inputs) else graph.eval reference inputs) byte := by
  rw [stored.retained byte,api.store _ _ _ _ _ stored.store byte,stored.execution.reference_value api ports]

#axiomsClean Execution.native_value
#axiomsClean Execution.reference_value
#axiomsClean StoredExecution.reference_bytes
#specNonCircular PrimitiveContracts avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
