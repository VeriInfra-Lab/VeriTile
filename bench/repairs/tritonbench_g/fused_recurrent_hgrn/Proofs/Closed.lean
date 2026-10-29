import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.NativeHost
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.ComputedGradient
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Physical

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Closed
open VeriTile.Triton FP.Structural
open Checked
noncomputable section
variable {α : Type} [Inhabited α]

def entry (memory : RegionName → Nat → α) : State α where
  mem := fun r a => .mk .real (memory r a)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def input (memory : RegionName → Nat → α) (T D : Nat) : Input α := fun r bh t d =>
  memory r (if r="h0" ∨ r="dht" then stateAddr bh D d else rowAddr bh T D t d)

theorem entry_inputs (memory : RegionName → Nat → α) (T D : Nat) :
    Execution.Inputs (entry memory) (input memory T D) T D := by
  constructor
  · intro r hr bh t d
    rcases hr with rfl|rfl|rfl <;> simp [Execution.read,entry,input,Cell.read]
  · intro r hr bh d
    rcases hr with rfl|rfl <;> simp [Execution.read,entry,input,Cell.read]

/-- Total closed interpreters. The theorems below derive that the fallback is
unreachable from the complete typed initialization/loop/postlude/grid proofs. -/
def first (p : Ops α) (c : Shape) (BD : Nat) (U S : Bool) (memory : RegionName → Nat → α) : State α :=
  (Execution.gridRun p (forward c.T c.D BD U S).surfaceBody
    (Execution.programs c.BH c.D BD) (entry memory)).getD (entry memory)

def second (p : Ops α) (c : Shape) (BDF BDB : Nat) (U S hasDo hasDht : Bool)
    (memory : RegionName → Nat → α) (incoming finalIncoming : Nat → α) : State α :=
  let f := first p c BDF U S memory
  let prepared := Execution.installCotangents p f hasDo incoming finalIncoming
  (Execution.gridRun p (backward c.T c.D BDB U (S && hasDht)).surfaceBody
    (Execution.programs c.BH c.D BDB) prepared).getD prepared

theorem first_complete (p : Ops α) (c : Shape) (BD : Nat) (positive : 0<BD) (U S : Bool)
    (memory : RegionName → Nat → α) :
    Execution.gridRun p (forward c.T c.D BD U S).surfaceBody
        (Execution.programs c.BH c.D BD) (entry memory)=some (first p c BD U S memory) ∧
    Execution.Saved p (first p c BD U S memory) (input memory c.T c.D) U c.BH c.T c.D ∧
    (S=Bool.true → ∀ bh,bh<c.BH → ∀ d,d<c.D →
      (first p c BD U S memory).mem "ht" (stateAddr bh c.D d)=
        .mk .real (state p U (input memory c.T c.D) bh d c.T)) ∧
    (∀ r a,¬Execution.ForwardRegion c.BH c.T c.D S r a →
      (first p c BD U S memory).mem r a=(entry memory).mem r a) := by
  obtain ⟨f,success,_,values,finalState,frame⟩ := Execution.forward_grid p (input memory c.T c.D)
    c.BH c.T c.D BD positive U S (entry memory) (entry_inputs memory c.T c.D)
  have actual : first p c BD U S memory=f := by simp [first,success]
  rw [actual]
  exact ⟨success,values,finalState,frame⟩

theorem complete (p : Ops α) (c : Shape) (BDF BDB : Nat) (positiveF : 0<BDF) (positiveB : 0<BDB)
    (U S hasDo hasDht : Bool) (memory : RegionName → Nat → α) (incoming finalIncoming : Nat → α) :
    let f := first p c BDF U S memory
    let b := second p c BDF BDB U S hasDo hasDht memory incoming finalIncoming
    let data := Execution.hostInput p (input memory c.T c.D) c.T c.D hasDo incoming finalIncoming
    Execution.gridRun p (backward c.T c.D BDB U (S && hasDht)).surfaceBody
      (Execution.programs c.BH c.D BDB) (Execution.installCotangents p f hasDo incoming finalIncoming)=some b ∧
    Execution.Saved p b (input memory c.T c.D) U c.BH c.T c.D ∧
    Execution.Gradients p b data U (S && hasDht) c.BH c.T c.D ∧
    (S=Bool.true → ∀ bh,bh<c.BH → ∀ d,d<c.D →
      b.mem "ht" (stateAddr bh c.D d)=.mk .real (state p U (input memory c.T c.D) bh d c.T)) ∧
    (∀ r a,r≠("do" : RegionName) → r≠("dht" : RegionName) →
      ¬Execution.ForwardRegion c.BH c.T c.D S r a → ¬Execution.BackwardRegion c.BH c.T c.D r a →
      b.mem r a=(entry memory).mem r a) := by
  obtain ⟨f,b,hf,hb,values,gradients,finalState,frame⟩ := Execution.host_result p (input memory c.T c.D)
    c.BH c.T c.D BDF BDB positiveF positiveB U S hasDo hasDht incoming finalIncoming
      (entry memory) (entry_inputs memory c.T c.D)
  have actualF : first p c BDF U S memory=f := by simp [first,hf]
  have actualB : second p c BDF BDB U S hasDo hasDht memory incoming finalIncoming=b := by
    simp only [second,actualF,hb,Option.getD_some]
  dsimp only
  rw [actualF,actualB]
  exact ⟨hb,values,gradients,finalState,frame⟩

#axiomsClean first_complete
#axiomsClean complete
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Closed
