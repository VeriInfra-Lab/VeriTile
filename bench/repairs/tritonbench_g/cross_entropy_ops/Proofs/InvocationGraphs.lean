import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ScalarGraph
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Source

namespace VeriTile.Bench.Repairs.CrossEntropyOps.InvocationGraphs
open CrossEntropy1.Floating (Word Numerics Obligations ScalarGraph)
noncomputable section

inductive Kind where
  | forward | backward | localLSE | localSum | rankLSE | rankSum | square | scale | add
  | runtime (operation : String)
  deriving DecidableEq

/-- All pointer properties affecting specialization are part of the request.
Shapes and strides do not replace the actual pointer's byte alignment. -/
structure PointerMetadata where
  role : String
  dtype : String
  address : Nat
  allocation : Nat
  storageOffsetBytes : Nat
  shape : List Nat
  strides : List Nat
  device : Nat
  isNegativeView : Bool
  deriving DecidableEq

/-- Complete per-invocation compiler/dispatch metadata. Scalar words retain
signed zero and NaN payloads. A request also retains its exact before-state;
there is deliberately no width-indexed DeviceFamily. -/
structure Request (σ : Type) where
  invocation : Nat
  before : σ
  sourceDigest : String
  implementation : String
  kind : Kind
  pointers : List PointerMetadata
  scalarWords : List Word
  scalarIntegers : List Int
  constexpr : List (String × Int)
  grid : List Nat
  blockWidth : Nat
  warps : Nat
  stages : Nat
  fusion : Bool
  target : String
  compilerVersion : String
  stream : Nat
  rank : Nat
  group : Option Nat
  work : Option Nat

/-- A selected graph belongs to one complete request and one loaded image.
This is only an interface for the still-required actual-call/image binding.
Its relation is structural, not a numerical conformance port. -/
structure Selection (σ : Type) where
  selected : (request : Request σ) → String → (width : Nat) → ScalarGraph width → Prop

structure Selected {σ : Type} (platform : Selection σ) (request : Request σ) (width : Nat) where
  imageDigest : String
  graph : ScalarGraph width
  binding : platform.selected request imageDigest width graph

/-- Reuse the established fixed-arity numerical induction on the graph chosen
for this request. This does not assert that an arbitrary graph is the image,
nor that the actual Python/native returned tensor has been connected to it. -/
theorem selected_graph_equiv {σ : Type} (platform : Selection σ) (request : Request σ)
    (width : Nat) (selected : Selected platform request width) (native reference : Numerics)
    (ports : Obligations native reference) (inputs : Fin width → Word) :
    selected.graph.eval native inputs=selected.graph.eval reference inputs :=
  selected.graph.equiv ports inputs inputs (fun _ => rfl)

/-- Interface consistency control only: the same block width can select
separate multiply/add and FMA graphs at different actual pointer alignments.
The concrete SM90 witness is separately recorded in offline002 (5 versus 2
FMA instructions for the two width-65536 forward images). -/
def alignmentControl (address : Nat) : ScalarGraph 3 :=
  if address%16=0 then .fma (.input ⟨0,by omega⟩) (.input ⟨1,by omega⟩) (.input ⟨2,by omega⟩)
  else .binary .add (.binary .mul (.input ⟨0,by omega⟩) (.input ⟨1,by omega⟩)) (.input ⟨2,by omega⟩)

theorem alignment_graphs_differ : alignmentControl 65536≠alignmentControl 65540 := by
  simp [alignmentControl]

/-- No simplification equates the two controls' numerical values. -/
theorem fused_operand_order (n : Numerics) (inputs : Fin 3 → Word) :
    (alignmentControl 65536).eval n inputs=n.fma (inputs ⟨0,by omega⟩) (inputs ⟨1,by omega⟩) (inputs ⟨2,by omega⟩) := by
  simp [alignmentControl,ScalarGraph.eval]

#axiomsClean selected_graph_equiv
#axiomsClean alignment_graphs_differ
#axiomsClean fused_operand_order
#specNonCircular Request avoiding [Source.forward,Source.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.InvocationGraphs
