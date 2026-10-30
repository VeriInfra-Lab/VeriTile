import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.ActualGrid
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.RuntimeObserved
import bench.tritonbench_g.rms_norm_triton.Proofs.Observed
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.RmsNormTriton.KernelObserved
open VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton FP.Structural Whole Observed
noncomputable section

/-- Successful complete source graph, independent native words, and exact
exterior typed-memory frame. Runtime epsilon and the existing primitive and
reduction-tree boundaries are explicit raw kernel parameters. -/
theorem kernel_native_equivalence (c : Layout) (eps : ℝ)
    (M : NativeModel (BitVec 32)) (epsilonWord : BitVec 32)
    (epsilonArgument : M.epsilon = epsilonWord) (profile : ReductionFamily)
    (xSeparate : c.X ≠ c.Y) (wSeparate : c.W ≠ c.Y)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M))
    (s : State (Term (BitVec 32))) (xs : Fin c.M → Nat → BitVec 32) (ws : Nat → BitVec 32)
    (input : ∀ row : Fin c.M, ∀ j, j < c.N →
      (s.mem c.X (row.val*c.N+j)).read .real = Term.input (xs row j))
    (weight : ∀ j, j < c.N → (s.mem c.W j).read .real = Term.input (ws j)) :
    M.epsilon = epsilonWord ∧
    ∃ final, ActualGrid.runRows graphAlgebra c (.parameter "eps") rows s = some final ∧
      (∀ row : Fin c.M, ∀ i : Fin c.N,
        interpret M profile i.val ((final.mem c.Y (row.val*c.N+i.val)).read .real) =
          RuntimeObserved.reference M profile c.N c.B (xs row) ws i.val) ∧
      (∀ region a, (region ≠ c.Y ∨ c.M*c.N ≤ a) → final.mem region a = s.mem region a) := by
  obtain ⟨final,ran,values,frame⟩ := ActualGrid.grid_result graphAlgebra c (.parameter "eps") xSeparate wSeparate rows schedule s
  refine ⟨epsilonArgument,final,ran,?_,frame⟩
  intro row i
  rw [values row i,Cell.read_same]
  change interpret M profile i.val (ActualRow.rowValue graphAlgebra c (.parameter "eps") _ _ i.val) = _
  rw [ActualRow.rowValue_congr graphAlgebra c (.parameter "eps") _ _ (fun j => Term.input (xs row j))
    (fun j => Term.input (ws j)) i.val i.isLt (input row) weight]
  exact RuntimeObserved.source_graph_lowering M profile c (xs row) ws i.val

#guard_msgs (drop info) in
#specNonCircular RuntimeObserved.reference avoiding [kernel,lower,interpret,rowValue,sourceRun,runRows]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsNormTriton.KernelObserved
