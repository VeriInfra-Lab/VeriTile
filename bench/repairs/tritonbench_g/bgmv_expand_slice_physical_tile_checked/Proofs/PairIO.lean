import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.PairConnection

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PairIO
open VeriTile Triton FP.Structural Direct Direct.FPValues PairConnection
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
variable {A : Type} [Inhabited A]

/-- Only the concrete kernel changes. The pids, argument binding, backend,
register clearing, windows and native observation are shared verbatim. -/
theorem machine_comm (M : Algebra A) (R : Reduction A) (comm : AdditionCommutes M) :
    AdditionCommutes (machine M R) := by
  simpa [AdditionCommutes,machine,Algebra.withDefaultPrecision,resolvePrecision] using comm

theorem invocation_run (M : Algebra A) (R : Reduction A) (h : Call)
    (comm : AdditionCommutes M) (p : Geometry.Program h) (s : State A) :
    (PublicIO.invocation M R h p).run s = (invocation M R h p).run s := by
  exact (source_pair (machine M R) (machine_comm M R comm) h).run _

theorem invocation_safe (M : Algebra A) (R : Reduction A) (h : Call)
    (comm : AdditionCommutes M) (p : Geometry.Program h) (bounds : RegionName → Nat) (s : State A) :
    (PublicIO.invocation M R h p).Safe bounds s ↔ (invocation M R h p).Safe bounds s := by
  exact (source_pair (machine M R) (machine_comm M R comm) h).safe bounds _

theorem schedule_run (M : Algebra A) (R : Reduction A) (h : Call)
    (comm : AdditionCommutes M) (ps : List (Geometry.Program h)) (s : State A) :
    runSchedule (ps.map (PublicIO.invocation M R h)) s =
      runSchedule (ps.map (invocation M R h)) s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.map_cons,runSchedule,invocation_run M R h comm]
    cases (invocation M R h p).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safe (M : Algebra A) (R : Reduction A) (h : Call)
    (comm : AdditionCommutes M) (ps : List (Geometry.Program h)) (bounds : RegionName → Nat) (s : State A) :
    ScheduleSafe bounds (ps.map (PublicIO.invocation M R h)) s ↔
      ScheduleSafe bounds (ps.map (invocation M R h)) s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_safe M R h comm,invocation_run M R h comm]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Complete actual two-program equivalence. Safety, successful execution,
all active tagged outputs and both exterior frames are conclusions. -/
theorem equivalent (M : Algebra A) (R : Reduction A) (h : Call) (comm : AdditionCommutes M) :
    (PublicIO.sourceIO M R id h).EquivalentOn (candidate M R h) PublicIO.Valid := by
  intro xs valid bounds hb s hin
  obtain ⟨safe,final,ran,values,frame⟩ := PublicIO.source_implements M R id h xs valid bounds hb s hin
  refine ⟨safe,(schedule_safe M R h comm _ bounds s).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run M R h comm _ s).symm.trans ran
  · intro o j _
    rfl

/-- Fix just the FP32 addition primitive to the existing strict software
model. All other local operations and the reduction plan remain explicit and
unchanged; no formula or kernel execution is a field of this model. -/
def strictAlgebra (M : Algebra (FP.Value .fp32)) (cfg : FP.Config) : Algebra (FP.Value .fp32) :=
  { M with binary := fun p d op a b =>
    if p=some .fp32 ∧ d=.real ∧ op=.add then FP.add cfg a b else M.binary p d op a b }

theorem strict_addition (M : Algebra (FP.Value .fp32)) (cfg : FP.Config) :
    AdditionCommutes (strictAlgebra M cfg) := by
  intro a b
  simpa [strictAlgebra] using FP.ScalarOpsLaws.add_comm cfg a b

#axiomsClean equivalent
#axiomsClean strict_addition
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PairIO
