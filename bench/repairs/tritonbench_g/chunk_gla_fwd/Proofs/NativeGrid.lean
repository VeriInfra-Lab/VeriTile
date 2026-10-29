import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeProgram
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeFrame

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeGrid
open VeriTile Triton FP.Structural
open FPStorage (Memory state payload)
open FPStages (Stage)
open NativeReads
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

def ReadCompatible (observe : RegionName → Cell α → α) (stage : Stage) : Prop :=
  ∀ port∈NativeLaunch.inputExtents stage,∀ cell,
    if port.1="h" then observe port.1 cell=cell.read .fp32
    else observe port.1 cell=cell.read .real

def Covers (scope : RegionName → Nat → Prop) (stage : Stage) : Prop :=
  ∀ port∈NativeLaunch.inputExtents stage,∀ off,off<port.2 → scope port.1 off

theorem bound_reads (scope : RegionName → Nat → Prop) (observe : RegionName → Cell α → α)
    (stage : Stage) (covers : Covers scope stage) (compatible : ReadCompatible observe stage)
    (s : State α) (m : Memory α) (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off) :
    Reads stage s m := by
  intro port member off within
  have value := hm port.1 off (covers port member off within)
  have reading := compatible port member (s.mem port.1 off)
  by_cases history : port.1="h"
  · simp only [if_pos history] at reading ⊢
    exact reading.symm.trans value
  · simp only [if_neg history] at reading ⊢
    exact reading.symm.trans value

theorem jobs_reference (scope : RegionName → Nat → Prop) (observe : RegionName → Cell α → α)
    (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (covers : Covers scope stage) (reads : ReadCompatible observe stage)
    (writes : NativeProgram.WriteCompatible observe stage)
    (order : List (GridIndex stage.grid)) (s : State α) (m : Memory α)
    (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off) :
    ∃ t,runFillSchedule (order.map (fun idx=>.source (NativeLaunch.job M D stage idx))) s=some t ∧
      (∀ r off,scope r off → observe r (t.mem r off)=
        FPGrid.referenceJobs (FPStages.referenceStep M D stage) order m r off) ∧
      (∀ r off,r≠stage.region ∨ stage.size≤off → t.mem r off=s.mem r off) := by
  induction order generalizing s m with
  | nil => exact ⟨s,rfl,hm,fun _ _ _=>rfl⟩
  | cons idx rest ih =>
    obtain ⟨next,ran,observed⟩ := NativeProgram.source_reference scope observe M D stage idx s m hm
      (bound_reads scope observe stage covers reads s m hm) writes
    obtain ⟨final,continued,values,frame⟩ := ih next (FPStages.referenceStep M D stage idx m) observed
    refine ⟨final,?_,values,?_⟩
    · simpa only [List.map_cons,runFillSchedule,FillStage.run,ran,Option.bind_some] using continued
    · intro r off outside
      exact (frame r off outside).trans (NativeFrame.source_frame M D stage idx s next ran r off outside)

theorem grid_reference (scope : RegionName → Nat → Prop) (observe : RegionName → Cell α → α)
    (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (covers : Covers scope stage) (reads : ReadCompatible observe stage)
    (writes : NativeProgram.WriteCompatible observe stage)
    (s : State α) (m : Memory α)
    (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off) :
    ∃ t,runFillSchedule (NativeLaunch.jobs M D stage) s=some t ∧
      (∀ r off,scope r off → observe r (t.mem r off)=FPStages.reference M D stage m r off) ∧
      (∀ r off,r≠stage.region ∨ stage.size≤off → t.mem r off=s.mem r off) :=
  jobs_reference scope observe M D stage covers reads writes (FPGrid.jobs stage.grid) s m hm

theorem run_append (xs ys : List (FillStage α)) (s : State α) :
    runFillSchedule (xs++ys) s=(runFillSchedule xs s).bind (runFillSchedule ys) := by
  induction xs generalizing s with
  | nil => rfl
  | cons first rest ih =>
    simp only [List.cons_append,runFillSchedule]
    cases first.run s <;> simp [ih]

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeGrid
