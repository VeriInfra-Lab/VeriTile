import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeRuntime
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FiniteSource

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.CheckedHost
open VeriTile Triton
open NativeInstructions
noncomputable section
set_option maxHeartbeats 8000000

/-- The identity and metadata of the object produced by new_zeros/empty_like.
These are ordinary allocator facts; no source execution or numeric value is a field. -/
structure Allocation (r : Admission.Request) (c : Host.Dimensions) where
  objectId : Nat
  tensor : Admission.Tensor
  abi : Physical.ABI c
  fresh : NativeRuntime.Fresh r.mode abi
  tensorValid : tensor.Valid r.query.device
    (match r.mode with
     | .producer => [c.B,c.H,c.T,c.BT]
     | .consumer => [c.B,c.H,c.T,c.V])
  pointer : tensor.byteBase=4*(abi.alloc (match r.mode with | .producer=>"attention" | .consumer=>"o")).base

/-- Setup receives the owner device and selects only the diagonal reduction
configuration. Inter/output warp choices do not alter a reduction tree. -/
structure Runtime (r : Admission.Request) (c : Host.Dimensions) where
  allocate : Except String (Allocation r c)
  setup : Allocation r c → Nat → Except String (Fin 4)

inductive Stage where | admission | allocation | outputGuard | setup | execution deriving DecidableEq
structure Failure where
  stage : Stage
  currentDevice : Nat
structure Returned where
  objectId : Nat
  tensor : Admission.Tensor
  memory : Nat → Graph
  currentDevice : Nat
  launchDevice : Option Nat

/-- Private allocations and unsuccessful setup do not publish a tensor. The
observable memory on those exits is the caller's live allocations. Successful
source execution is computed by NativeRuntime.execute, never returned by setup. -/
def runAdmitted (r : Admission.Request) (a : {c : Host.Dimensions // Admission.Contract r c})
    (rt : Runtime r a.val) (before : Nat → BitVec 32) (caller : Nat) : Except Failure Returned :=
  match rt.allocate with
  | .error _ => .error ⟨.allocation,caller⟩
  | .ok allocation =>
    if Admission.empty r.mode a.val=Bool.true then
      match NativeRuntime.execute r.mode 0 r.scale.runtimeWord allocation.abi before with
      | none => .error ⟨.execution,caller⟩
      | some memory => .ok ⟨allocation.objectId,allocation.tensor,memory,caller,none⟩
    else if allocation.tensor.AddressValid a.val then
      match rt.setup allocation r.query.device with
      | .error _ => .error ⟨.setup,caller⟩
      | .ok w =>
        match NativeRuntime.execute r.mode w r.scale.runtimeWord allocation.abi before with
        | none => .error ⟨.execution,caller⟩
        | some memory => .ok ⟨allocation.objectId,allocation.tensor,memory,caller,some r.query.device⟩
    else .error ⟨.outputGuard,caller⟩

def ResultSpec (r : Admission.Request) (a : {c : Host.Dimensions // Admission.Contract r c})
    (rt : Runtime r a.val) (before : Nat → BitVec 32) (caller : Nat) : Except Failure Returned → Prop
  | .error error => error.stage≠.execution ∧ error.currentDevice=caller
  | .ok result => ∃ allocation w,rt.allocate=.ok allocation ∧
      result.objectId=allocation.objectId ∧ result.tensor=allocation.tensor ∧
      result.currentDevice=caller ∧
      result.launchDevice=(if Admission.empty r.mode a.val=Bool.true then none else some r.query.device) ∧
      NativeRuntime.execute r.mode w r.scale.runtimeWord allocation.abi before=some result.memory ∧
      result.memory=NativeRuntime.reference r.mode w r.scale.runtimeWord allocation.abi before ∧
      (∀ p,NativeRuntime.Outside r.mode allocation.abi p → result.memory p=.bits (before p)) ∧
      (∀ region,Physical.tracked region → NativeRuntime.ReadOnly r.mode region →
        ∀ i,i<(allocation.abi.alloc region).length →
          result.memory ((allocation.abi.alloc region).base+i)=.bits (before ((allocation.abi.alloc region).base+i)))

theorem complete (r : Admission.Request) (a : {c : Host.Dimensions // Admission.Contract r c})
    (rt : Runtime r a.val) (before : Nat → BitVec 32) (caller : Nat) :
    ResultSpec r a rt before caller (runAdmitted r a rt before caller) := by
  unfold runAdmitted
  cases allocated : rt.allocate with
  | error reason => simp [ResultSpec]
  | ok allocation =>
    simp only
    by_cases empty : Admission.empty r.mode a.val=Bool.true
    · rw [if_pos empty,NativeRuntime.source_complete]
      exact ⟨allocation,0,allocated,rfl,rfl,rfl,by simp [empty],
        NativeRuntime.source_complete _ _ _ _ _,rfl,NativeRuntime.frame _ _ _ _ _,NativeRuntime.inputs _ _ _ _ allocation.fresh _⟩
    · rw [if_neg empty]
      split
      · cases setup : rt.setup allocation r.query.device with
        | error reason => simp [ResultSpec]
        | ok w =>
          simp only
          rw [NativeRuntime.source_complete]
          exact ⟨allocation,w,allocated,rfl,rfl,rfl,by simp [empty],
            NativeRuntime.source_complete _ _ _ _ _,rfl,NativeRuntime.frame _ _ _ _ _,NativeRuntime.inputs _ _ _ _ allocation.fresh _⟩
      · exact ⟨by simp,rfl⟩

/-- Complete metadata rejection occurs before allocation or CUDA setup. -/
def run (r : Admission.Request)
    (rt : ∀ a : {c : Host.Dimensions // Admission.Contract r c},Runtime r a.val)
    (before : Nat → BitVec 32) (caller : Nat) : Except Failure Returned :=
  match Admission.admit r with
  | .error _ => .error ⟨.admission,caller⟩
  | .ok a => runAdmitted r a (rt a) before caller

theorem rejected (r : Admission.Request)
    (rt : ∀ a : {c : Host.Dimensions // Admission.Contract r c},Runtime r a.val)
    (before : Nat → BitVec 32) (caller : Nat) (why : String)
    (bad : Admission.admit r=.error why) : run r rt before caller=.error ⟨.admission,caller⟩ := by
  simp [run,bad]

theorem wrapper_complete (r : Admission.Request)
    (rt : ∀ a : {c : Host.Dimensions // Admission.Contract r c},Runtime r a.val)
    (before : Nat → BitVec 32) (caller : Nat) :
    match Admission.admit r with
    | .error _ => run r rt before caller=.error ⟨.admission,caller⟩
    | .ok a => ResultSpec r a (rt a) before caller (run r rt before caller) := by
  cases check : Admission.admit r with
  | error why => simp [run,check]
  | ok a => simpa [run,check] using complete r a (rt a) before caller

#axiomsClean complete
#axiomsClean wrapper_complete
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.CheckedHost
