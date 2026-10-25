import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.LegacySourcePreservation
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.ActualGrid
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.RuntimeObserved

/-! The complete current source is executed directly in ActualRow/ActualGrid.
Historical literal/cast identities remain explicitly in LegacyPastedSurface.
The caller supplies the epsilon value, not a synthetic statement in the body. -/
namespace VeriTile.Bench.Repairs.RmsNormTriton.SourcePreservation
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Every memory cell agrees after moving the earlier scalar ABI conversion
from the body into its actual call argument. This holds for arbitrary M.cast;
it does not assert identity of a same-precision cast. -/
theorem row_memory {α : Type} [Inhabited α] (M : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) :
    (ActualRow.sourceRun M c (ActualRow.argument M eps) pid s).val.mem =
      (Whole.sourceRun M c eps pid s).val.mem := by
  funext region a
  by_cases same : region = c.Y
  · subst region
    by_cases active : pid*c.N ≤ a ∧ a < pid*c.N+c.N
    · have small : a-pid*c.N < c.N := by omega
      let i : Fin c.B := ⟨a-pid*c.N,small.trans_le c.complete⟩
      have eq : a=pid*c.N+i.val := by dsimp [i]; omega
      rw [eq,ActualRow.row_readback M c (ActualRow.argument M eps) pid s i small,
        Whole.row_readback M c eps pid s i small,ActualRow.rowValue_argument]
    · have outside : Whole.Outside c pid c.Y a := by
        right
        intro i hi equal
        apply active
        rw [equal]
        exact ⟨Nat.le_add_right _ _,Nat.add_lt_add_left hi _⟩
      rw [ActualRow.row_frame M c (ActualRow.argument M eps) pid s c.Y a outside,
        Whole.row_frame M c eps pid s c.Y a outside]
  · rw [ActualRow.row_frame M c (ActualRow.argument M eps) pid s region a (Or.inl same),
      Whole.row_frame M c eps pid s region a (Or.inl same)]

/-- Both old/new complete call executions succeed and have the same full
memory observation; the actual call also initializes its scalar register. -/
theorem row_execution {α : Type} [Inhabited α] (M : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) :
    (run (FP.ProfiledRewrite.engine M) (ActualRow.source c).surfaceBody
      (ActualRow.entry (ActualRow.argument M eps) pid s)).map State.mem =
    (run (FP.ProfiledRewrite.engine M) (c.source eps).surfaceBody (rowState pid s)).map State.mem := by
  rw [(ActualRow.sourceRun M c (ActualRow.argument M eps) pid s).property,
    (Whole.sourceRun M c eps pid s).property]
  exact congrArg some (row_memory M c eps pid s)

/-- The current native epsilon is an actual named argument. Both mathematical
operation graphs retain the previous independent native instruction recurrence. -/
theorem native_row_observation (M : Observed.NativeModel α)
    (profile : Observed.ReductionFamily) (c : Layout) (eps : ℝ)
    (xs ws : Nat → α) (i : Nat) :
    Observed.interpret M profile i (ActualRow.rowValue Observed.graphAlgebra c
      (.parameter "eps") (fun j => .input (xs j)) (fun j => .input (ws j)) i) =
    Observed.interpret M profile i (Whole.rowValue Observed.graphAlgebra c eps
      (fun j => .input (xs j)) (fun j => .input (ws j)) i) := by
  rw [RuntimeObserved.source_graph_lowering,Observed.source_graph_lowering]

#axiomsClean row_memory
#axiomsClean row_execution
#axiomsClean native_row_observation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsNormTriton.SourcePreservation
