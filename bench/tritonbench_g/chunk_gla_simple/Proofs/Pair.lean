import bench.tritonbench_g.chunk_gla_simple.PairIO
import bench.tritonbench_g.chunk_gla_simple.Proofs.PairSource
import bench.tritonbench_g.chunk_gla_simple.Proofs.IONative
import VeriTile.Triton.Float.ObservedProgramEquivalence

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PairIO
open VeriTile Triton FP.Structural Host
open scoped VeriTile.Spec
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

theorem launch_run (c : Shape) (q : GridIndex (grid c)) (s : State NativeGraph.Expr) :
    (candidateLaunch c q).run s = (PublicIO.nativeLaunch c q).run s := by
  exact PairSource.execution_eq _ c ScalarGraph.scaleTag _

theorem launch_safe (c : Shape) (q : GridIndex (grid c)) (s : State NativeGraph.Expr)
    (bounds : RegionName → Nat) :
    (candidateLaunch c q).Safe bounds s ↔ (PublicIO.nativeLaunch c q).Safe bounds s := by
  exact PairSource.safe_iff _ c ScalarGraph.scaleTag _ bounds

theorem schedule_run (c : Shape) (jobs : List (GridIndex (grid c)))
    (s : State NativeGraph.Expr) :
    runSchedule (jobs.map (candidateLaunch c)) s =
      runSchedule (jobs.map (PublicIO.nativeLaunch c)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    simp only [List.map_cons,runSchedule,launch_run]
    cases (PublicIO.nativeLaunch c q).run s <;> simp only [Option.bind,ih]

theorem schedule_safe (c : Shape) (jobs : List (GridIndex (grid c)))
    (s : State NativeGraph.Expr) (bounds : RegionName → Nat) :
    ScheduleSafe bounds (jobs.map (candidateLaunch c)) s ↔
      ScheduleSafe bounds (jobs.map (PublicIO.nativeLaunch c)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    simp only [List.map_cons,ScheduleSafe,launch_safe,launch_run,ih]

theorem candidate_implements (c : Shape) (bt : 0<c.BT) :
    (candidate c).toProgramIO.Implements (PublicNative.expressionFormula c bt) := by
  intro values _ bounds backed initial bound
  obtain ⟨safe,final,executed,outputs,frame⟩ :=
    PublicIO.native_implements c bt values trivial bounds backed initial bound
  refine ⟨?_,final,?_,outputs,frame⟩
  · exact (schedule_safe c (FPGrid.programs c) initial bounds).mpr safe
  · exact (schedule_run c (FPGrid.programs c) initial).trans executed

/-- Complete safe, successful schedules and identical typed outputs/frames on
the original arbitrary symbolic domain, before any native output observation. -/
theorem equivalent (c : Shape) (bt : 0<c.BT) :
    (PublicIO.nativeIO c).Equivalent (candidate c) :=
  ProgramIO.EquivalentOn.of_implementsOn
    (PublicIO.native_implements c bt) (candidate_implements c bt)

/-- The original scalar-FMA/exp2/runtime-scale observation remains identical
for every scale word. No input is restricted to a leaf or finite value. -/
theorem observed_equivalent (c : Shape) (bt : 0<c.BT) (scale : BitVec 32) :
    (PublicIO.nativeIO c).ObservedEquivalent (candidate c)
      (fun _ _ => ScalarGraph.lower scale) :=
  ProgramIO.ObservedEquivalent.of_equivalent (equivalent c bt) _

theorem main (c : Shape) (bt : 0<c.BT) : original c ≡[[]] rowReused c :=
  ProgramIO.Implementation.of_equivalent (candidate c) (equivalent c bt)

#axiomsClean equivalent
#axiomsClean observed_equivalent
#axiomsClean main
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PairIO
