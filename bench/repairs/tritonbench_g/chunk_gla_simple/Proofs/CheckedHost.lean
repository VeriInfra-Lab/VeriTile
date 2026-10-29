import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.PhysicalDirect

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.CheckedHost
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions (Graph)
noncomputable section
set_option maxHeartbeats 8000000

/-- Ordinary empty_like metadata and freshness. Every input base is the
original checked tensor's actual data pointer; no input materialization occurs. -/
structure Allocation (r : Admission.Request) where
  layout : PhysicalDirect.Layout r.shape
  tensor : Admission.Tensor
  valid : tensor.Valid r.key.device [r.shape.B,r.shape.H,r.shape.T,r.shape.V]
  outputPointer : tensor.byteBase=4*layout.base "o"
  outputDevice : layout.outputDevice=r.key.device
  queryPointer : layout.base "q"=r.query.byteBase/4
  keyPointer : layout.base "k"=r.key.byteBase/4
  valuePointer : layout.base "v"=r.value.byteBase/4
  historyPointer : layout.base "h"=r.history.byteBase/4
  gatePointer : layout.base "g"=r.gate.byteBase/4

structure Runtime (r : Admission.Request) where
  allocate : Except String (Allocation r)
  setup : Allocation r → Nat → Except String Unit

inductive Stage where | admission | allocation | outputGuard | setup deriving DecidableEq
structure Failure where
  stage : Stage
  currentDevice : Nat
structure Returned where
  objectId : Nat
  tensor : Admission.Tensor
  memory : Nat → Graph
  currentDevice : Nat
  launchDevice : Option Nat

def produced (r : Admission.Request) (a : Allocation r) (before : Nat → BitVec 32)
    (scratch : State NativeGraph.Expr) (caller : Nat) (launched : Bool) : Returned :=
  ⟨a.layout.objectId,a.tensor,
    PhysicalDirect.graphResult r.shape a.layout r.scale.runtimeWord before scratch,
    caller,if launched then some r.key.device else none⟩

/-- The empty branch returns immediately after allocation. Setup can fail for
unsupported dot shapes or target resources; successful setup contains no source
state or numerical conclusion. All launches use the owner's device. -/
def runAdmitted (r : Admission.Request) (rt : Runtime r) (before : Nat → BitVec 32)
    (scratch : State NativeGraph.Expr) (caller : Nat) : Except Failure Returned :=
  match rt.allocate with
  | .error _ => .error ⟨.allocation,caller⟩
  | .ok a =>
    if r.shape.heads*r.shape.T*r.shape.V=0 then .ok (produced r a before scratch caller Bool.false)
    else if Admission.Address a.tensor then
      match rt.setup a r.key.device with
      | .error _ => .error ⟨.setup,caller⟩
      | .ok _ => .ok (produced r a before scratch caller Bool.true)
    else .error ⟨.outputGuard,caller⟩

def run (r : Admission.Request) (rt : Runtime r) (before : Nat → BitVec 32)
    (scratch : State NativeGraph.Expr) (caller : Nat) : Except Failure Returned :=
  if Admission.check r then runAdmitted r rt before scratch caller else .error ⟨.admission,caller⟩

def ResultSpec (r : Admission.Request) (rt : Runtime r) (before : Nat → BitVec 32)
    (scratch : State NativeGraph.Expr) (caller : Nat) : Except Failure Returned → Prop
  | .error failure => failure.currentDevice=caller
  | .ok result => ∃ a,rt.allocate=.ok a ∧ result.objectId=a.layout.objectId ∧
      result.tensor=a.tensor ∧ result.currentDevice=caller ∧
      result.launchDevice=(if r.shape.heads*r.shape.T*r.shape.V=0 then none else some r.key.device) ∧
      result.memory=PhysicalDirect.graphResult r.shape a.layout r.scale.runtimeWord before scratch ∧
      (r.shape.heads*r.shape.T*r.shape.V≠0 → Admission.Address a.tensor ∧ rt.setup a r.key.device=.ok ()) ∧
      (∀ p,¬(PhysicalDirect.output a.layout).Contains p → result.memory p=.bits (before p)) ∧
      (∀ region∈(["q","k","v","h","g"] : List RegionName),∀ i,i<capacity r.shape region →
        result.memory (a.layout.base region+i)=.bits (before (a.layout.base region+i)))

theorem complete (r : Admission.Request) (rt : Runtime r) (before : Nat → BitVec 32)
    (scratch : State NativeGraph.Expr) (caller : Nat) :
    ResultSpec r rt before scratch caller (runAdmitted r rt before scratch caller) := by
  unfold runAdmitted
  cases allocation : rt.allocate with
  | error reason => rfl
  | ok a =>
    simp only
    by_cases empty : r.shape.heads*r.shape.T*r.shape.V=0
    · rw [if_pos empty]
      refine ⟨a,allocation,rfl,rfl,rfl,by simp [produced,empty],rfl,?_,
        PhysicalDirect.graph_frame _ _ _ _ _,PhysicalDirect.graph_inputs _ _ _ _ _⟩
      exact fun nonempty=>False.elim (nonempty empty)
    · rw [if_neg empty]
      split
      · rename_i address
        cases setup : rt.setup a r.key.device with
        | error reason => rfl
        | ok token =>
          cases token
          exact ⟨a,allocation,rfl,rfl,rfl,by simp [produced,empty],rfl,
            fun _=>⟨address,setup⟩,PhysicalDirect.graph_frame _ _ _ _ _,PhysicalDirect.graph_inputs _ _ _ _ _⟩
      · rfl

theorem wrapper_complete (r : Admission.Request) (rt : Runtime r) (before : Nat → BitVec 32)
    (scratch : State NativeGraph.Expr) (caller : Nat) :
    (Admission.check r=Bool.false → run r rt before scratch caller=.error ⟨.admission,caller⟩) ∧
    ResultSpec r rt before scratch caller (run r rt before scratch caller) := by
  constructor
  · intro rejected
    simp [run,rejected]
  · unfold run
    split
    · exact complete r rt before scratch caller
    · rfl

/-- On an admitted, nonempty success the finite source ABI is a consequence
of executable guards, independent of setup or the returned values. -/
theorem native_abi (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (nonempty : r.shape.heads*r.shape.T*r.shape.V≠0) (a : Allocation r)
    (before : Nat → BitVec 32) (scratch : State NativeGraph.Expr) :
    (PhysicalDirect.graphSource r.shape (PhysicalDirect.fpInput a.layout before scratch)).mem=
      (NativeABI.grid ScalarGraph.algebra r.shape (ScalarGraph.dots r.shape)
        (Admission.accepted r checked nonempty) ScalarGraph.scaleTag (FPGrid.programs r.shape)
        (PhysicalDirect.fpInput a.layout before scratch)).mem :=
  PhysicalDirect.graph_native _ _ _

#axiomsClean wrapper_complete
#axiomsClean native_abi
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.CheckedHost
