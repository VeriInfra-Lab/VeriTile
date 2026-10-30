import bench.tritonbench_g.kldiv_triton.Proofs.PublicIO
import bench.tritonbench_g.kldiv_triton.Proofs.SourcePair

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.ProgramPair
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem schedule_run (m : Tile.Model α) (left right : ComputeKernel)
    (same : ∀ s,FP.Structural.exec m.alg right s=FP.Structural.exec m.alg left s) (work : List Nat) (s : State α) :
    runSchedule (work.map (IO.launch m right)) s=runSchedule (work.map (IO.launch m left)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons row rest ih =>
    change (FP.Structural.exec m.alg right _).bind _=(FP.Structural.exec m.alg left _).bind _
    rw [same]
    congr 1
    funext t
    exact ih t

theorem schedule_safe (m : Tile.Model α) (left right : ComputeKernel)
    (same : ∀ s,FP.Structural.exec m.alg right s=FP.Structural.exec m.alg left s)
    (safe : ∀ bounds s,TraceSafe m.alg bounds right s ↔ TraceSafe m.alg bounds left s)
    (work : List Nat) (bounds : RegionName → Nat) (s : State α)
    (original : ScheduleSafe bounds (work.map (IO.launch m left)) s) :
    ScheduleSafe bounds (work.map (IO.launch m right)) s := by
  induction work generalizing s with
  | nil => trivial
  | cons row rest ih =>
    refine ⟨(safe bounds _).mpr original.1,?_⟩
    intro t ran
    apply ih
    apply original.2 t
    exact (same _).symm.trans ran

/-- The entire repaired forward source, including zero-target guard and masked
chunk sums, is preserved across every complete accepted row schedule. -/
theorem forward (m : Tile.Model α) (rows sy st so N B : Nat) (flag : Bool) (red : Nat)
    (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=outputCount N B red) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    (IO.forward m rows sy st so N B flag red work).Equivalent
      (IO.forwardCandidate m rows sy st so N B flag red work) := by
  intro vals _ bounds covered initial pins
  have spec := PublicIO.forward m rows sy st so N B flag red positive unique work order
    (KernelInputs.forwardValue m N B flag red) (fun _ _ _ => rfl)
  obtain ⟨safe,final,ran,_,frame⟩ := spec vals trivial bounds covered initial pins
  refine ⟨safe,schedule_safe m _ _ (SourcePair.forward_run m.alg sy st so N B flag red)
    (SourcePair.forward_safe m.alg sy st so N B flag red) work bounds initial safe,
    final,final,ran,?_,fun _ _ _ => rfl,frame,frame⟩
  exact (schedule_run m _ _ (SourcePair.forward_run m.alg sy st so N B flag red) work initial).trans ran

theorem backward (m : Tile.Model α) (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=N) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    (IO.backward m rows so st N B flag work).Equivalent (IO.backwardCandidate m rows so st N B flag work) := by
  intro vals _ bounds covered initial pins
  have spec := PublicIO.backward m rows so st N B flag positive unique work order
    (fun v => Tile.gradient m flag (m.alg.fp32Load v)) (fun _ => rfl)
  obtain ⟨safe,final,ran,_,frame⟩ := spec vals trivial bounds covered initial pins
  refine ⟨safe,schedule_safe m _ _ (SourcePair.backward_run m.alg so st N B flag)
    (SourcePair.backward_safe m.alg so st N B flag) work bounds initial safe,
    final,final,ran,?_,fun _ _ _ => rfl,frame,frame⟩
  exact (schedule_run m _ _ (SourcePair.backward_run m.alg so st N B flag) work initial).trans ran

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.ProgramPair
