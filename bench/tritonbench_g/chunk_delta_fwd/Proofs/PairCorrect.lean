import bench.tritonbench_g.chunk_delta_fwd.PairIO
import bench.tritonbench_g.chunk_delta_fwd.Proofs.PairSource
import bench.tritonbench_g.chunk_delta_fwd.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.PairCorrect
open VeriTile Triton FP.Structural FP.ProfiledRewrite ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph (WordPrimitives)
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
variable {α : Type} [Inhabited α]

theorem invocation_run (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID) (s : State α) :
    (Interface.invocation M D p pid).run s = (PairIO.invocation M D p pid).run s := by
  exact (PairSource.complete (engine (withDot M D)) p).runs _

theorem invocation_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (bounds : RegionName → Nat) (s : State α) :
    (Interface.invocation M D p pid).Safe bounds s ↔ (PairIO.invocation M D p pid).Safe bounds s := by
  exact (PairSource.complete (engine (withDot M D)) p).safe bounds _

theorem schedule_run (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (schedule : List PID) (s : State α) :
    runSchedule (schedule.map (Interface.invocation M D p)) s =
      runSchedule (schedule.map (PairIO.invocation M D p)) s := by
  induction schedule generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons, runSchedule, invocation_run]
    exact congrArg (fun f => ((PairIO.invocation M D p pid).run s).bind f) (funext ih)

theorem schedule_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (schedule : List PID)
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (schedule.map (Interface.invocation M D p)) s ↔
      ScheduleSafe bounds (schedule.map (PairIO.invocation M D p)) s := by
  induction schedule generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons, ScheduleSafe, invocation_run, invocation_safe]
    constructor <;> rintro ⟨head, tail⟩ <;> refine ⟨head, ?_⟩
    · intro t ht; exact (ih t).mp (tail t ht)
    · intro t ht; exact (ih t).mpr (tail t ht)

/-- Full permutation grid, source safety, successful runs, all outputs and both
exterior frames. Every input word and every native instruction port is retained. -/
theorem equivalent (I : WordPrimitives) (p : Parameters) (shape : Geometry.ShapeValid p)
    (schedule : List PID) (order : schedule.Perm (grid p)) :
    (PairIO.io I p schedule).Equivalent (PairIO.candidate I p schedule) := by
  intro xs _ bounds hb initial hin
  obtain ⟨safe, final, ran, out, frame⟩ :=
    IOImplements.native I p shape schedule order xs trivial bounds hb initial hin
  refine ⟨safe, ?_, final, final, ran, ?_, ?_, frame, frame⟩
  · exact (schedule_safe Native.algebra Native.dot p schedule bounds initial).mp safe
  · exact (schedule_run Native.algebra Native.dot p schedule initial).symm.trans ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.PairCorrect
