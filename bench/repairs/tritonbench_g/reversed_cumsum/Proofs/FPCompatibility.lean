import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPPrograms
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Rewrite
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPExecution

/-! Optional source-identity and complete execution/frame consequences.
The public equivalence itself needs no input or numerical guard. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.Equivalence
open VeriTile Triton FP.ProfiledRewrite
noncomputable section

theorem original_source (h : Call) (c : Config) : (original h c).kernel=Kernels.source h c := rfl

/-- A successful pair for any admitted program, arbitrary initial registers
and memory, and any total configured dot backend. -/
theorem successful_pair {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (dot : FPValues.DotBackend α) (h : Call) (distinct : h.input≠h.output)
    (c : Config) (chosen : c∈configurations) (row : Geometry.Row h) (feature : Geometry.FeatureBlock h)
    (u : FP.Structural.State α) (pid0 : u.pids 0=feature.val) (pid1 : u.pids 1=row.val) :
    ∃ v, FP.Structural.exec (FPValues.machine M dot) (original h c).kernel u=some v ∧
      FP.Structural.exec (FPValues.machine M dot) (compareRight h c).kernel u=some v ∧
      FPMemory.Frame h row.val feature.val u v := by
  obtain ⟨v,executed,_,frame⟩ := FPExecution.source_correct M dot h distinct c
    (Kernels.configurations_positive c chosen).1 row feature u pid0 pid1
  exact ⟨v,executed,(Rewrite.whole_state _ h c u).symm.trans executed,frame⟩

#axiomsClean original_source
#axiomsClean successful_pair
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsum.Equivalence
