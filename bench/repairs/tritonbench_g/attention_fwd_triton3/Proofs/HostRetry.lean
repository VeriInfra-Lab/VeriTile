import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Physical

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.HostRetry
open VeriTile.Triton Host Raw Physical
noncomputable section

structure Globals where
  bm : Nat
  bn : Nat
  deriving DecidableEq

def initial : Globals := ⟨64,64⟩
def half (g : Globals) : Globals := ⟨g.bm/2,g.bn/2⟩
def call (c : Call) (g : Globals) : Call := {c with BM:=g.bm,BN:=g.bn}

inductive Attempt where
  | success | outOfResources | otherException
  deriving DecidableEq

/-- The real wrapper catches only the first OutOfResources, mutates both
module globals once, and retries once. A second failure escapes with the
halved globals retained. No reset, clamp or per-call replacement is introduced. -/
def transition (g : Globals) (first second : Attempt) : Globals × Option Globals :=
  match first with
  | .success=>(g,some g)
  | .otherException=>(g,none)
  | .outOfResources=>
    let next:=half g
    (next,if second=.success then some next else none)

theorem normal (g : Globals) (second : Attempt) :
    transition g .success second=(g,some g) := rfl

theorem retry_failure_keeps_halved (g : Globals) (second : Attempt) (hs : second≠.success) :
    transition g .outOfResources second=(half g,none) := by simp [transition,hs]

theorem retry_persistent :
    transition initial .outOfResources .success=(⟨32,32⟩,some ⟨32,32⟩) ∧
    transition (half initial) .outOfResources .success=(⟨16,16⟩,some ⟨16,16⟩) := by decide

/-- Runtime launch/device/descriptor handling is an ordinary implementation
contract. A compile-time resource failure occurs before any kernel store;
therefore the selected launch sees the same six-input byte snapshot. -/
def invoke (c : Call) (g : Globals) (first second : Attempt) (A : FlatAlloc)
    (ops : Ops) (schedules : (config : Globals) → Schedule (call c config))
    (before : Bytes) (scale : Word) : Globals × Option Bytes :=
  let result:=transition g first second
  (result.1,result.2.map (fun config=>execute (call c config) A ops (schedules config) before scale))

theorem invoke_success (c : Call) (g : Globals) (second : Attempt) (A : FlatAlloc)
    (ops : Ops) (schedules : (config : Globals) → Schedule (call c config))
    (before : Bytes) (scale : Word) :
    invoke c g .success second A ops schedules before scale=
      (g,some (execute (call c g) A ops (schedules g) before scale)) := rfl

theorem invoke_retry (c : Call) (g : Globals) (A : FlatAlloc)
    (ops : Ops) (schedules : (config : Globals) → Schedule (call c config))
    (before : Bytes) (scale : Word) :
    invoke c g .outOfResources .success A ops schedules before scale=
      (half g,some (execute (call c (half g)) A ops (schedules (half g)) before scale)) := rfl

theorem invoke_equiv (c : Call) (g : Globals) (first second : Attempt) (A : FlatAlloc)
    (actual reference : Ops) (laws : LocalLaws actual reference)
    (schedules : (config : Globals) → Schedule (call c config)) (before : Bytes) (scale : Word) :
    invoke c g first second A actual schedules before scale=
      invoke c g first second A reference schedules before scale := by
  rw [operations_eq actual reference laws]

#print axioms retry_persistent
#print axioms invoke_equiv
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.HostRetry
