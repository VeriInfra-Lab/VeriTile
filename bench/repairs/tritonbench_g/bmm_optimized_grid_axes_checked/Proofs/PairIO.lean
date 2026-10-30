import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.PairConnection
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.IOComplete
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.PairIO
open VeriTile Triton FP.Structural BmmOptimized Interface
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 24000
variable {A : Type} [Inhabited A]

/-- The same three-dimensional launch and selected live-FMA graph backend;
only the complete quoted kernel changes its exact integer remainder. -/
theorem invocation_run (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (pid : Program) (initial : State (SourceGraph.Node A)) :
    (Interface.invocation s c p rest pid).run initial=(invocation s c p rest pid).run initial := by
  exact (PairConnection.source_pair (SourceGraph.engine p rest) (Source.parameters s c)).run _

theorem invocation_safe (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (pid : Program) (bounds : RegionName → Nat) (initial : State (SourceGraph.Node A)) :
    (Interface.invocation s c p rest pid).Safe bounds initial ↔
      (invocation s c p rest pid).Safe bounds initial := by
  exact (PairConnection.source_pair (SourceGraph.engine p rest) (Source.parameters s c)).safe bounds _

theorem schedule_run (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (ps : List Program) (initial : State (SourceGraph.Node A)) :
    runSchedule (ps.map (Interface.invocation s c p rest)) initial=
      runSchedule (ps.map (invocation s c p rest)) initial := by
  induction ps generalizing initial with
  | nil => rfl
  | cons pid ps ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    cases (invocation s c p rest pid).run initial with
    | none => rfl
    | some t => exact ih t

theorem schedule_safe (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (ps : List Program) (bounds : RegionName → Nat) (initial : State (SourceGraph.Node A)) :
    ScheduleSafe bounds (ps.map (Interface.invocation s c p rest)) initial ↔
      ScheduleSafe bounds (ps.map (invocation s c p rest)) initial := by
  induction ps generalizing initial with
  | nil => rfl
  | cons pid ps ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Full actual pair for every complete selected three-axis program order.
Both runs succeed safely with identical tagged observations and raw frames. -/
theorem equivalent (s : Shape) (c : Config) (p : Scalar A) (rest : Algebra A)
    (ps : List Program) (order : ps.Perm (programs s c)) :
    (Interface.io s c p rest ps).Equivalent (candidate s c p rest ps) := by
  intro vals _ bounds hb initial hin
  obtain ⟨safe,final,ran,out,frame⟩ := Interface.complete s c p rest ps order vals trivial bounds hb initial hin
  refine ⟨safe,(schedule_safe s c p rest ps bounds initial).mp safe,
    final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run s c p rest ps initial).symm.trans ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.PairIO
