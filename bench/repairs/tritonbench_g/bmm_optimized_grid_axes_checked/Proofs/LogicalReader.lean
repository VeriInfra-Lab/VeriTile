import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.PublicResults
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.IO

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
open VeriTile Triton FP.Structural BmmOptimized
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section


theorem logical_iff (s : Shape) (c : Config) (rest : Algebra ℝ) (order : List Program) :
    (logicalIO s c rest order ⊨ fun xs batch row col =>
      ∑ k ∈ Finset.range s.k, xs.aAt batch.val row.val k * xs.bAt batch.val k col.val) ↔
    (realIO s c rest order).Implements (realFormula s) := by
  change (realIO s c rest order).Implements
    (fun vals => (logicalOutputsEquiv s).symm (fun batch row col =>
      ∑ k ∈ Finset.range s.k, (logicalInputsEquiv s vals).aAt batch.val row.val k *
        (logicalInputsEquiv s vals).bAt batch.val k col.val)) ↔ _
  have eq : (fun vals => (logicalOutputsEquiv s).symm (fun batch row col =>
      ∑ k ∈ Finset.range s.k, (logicalInputsEquiv s vals).aAt batch.val row.val k *
        (logicalInputsEquiv s vals).bAt batch.val k col.val)) = realFormula s := by
    funext vals o
    fin_cases o
    funext j
    simp only [LogicalInputs.aAt,LogicalInputs.bAt,Equiv.symm_apply_apply]
    change (∑ k ∈ Finset.range s.k,
      arrays 0 s vals .a (s.a (j.1.val/s.n/s.m) (j.1.val/s.n%s.m) k) *
      arrays 0 s vals .b (s.b (j.1.val/s.n/s.m) k (j.1.val%s.n))) = _
    simp only [realFormula,mathematical,outputCell,Nat.div_div_eq_div_mul,Nat.mul_comm]
  exact (congrArg (fun f => (realIO s c rest order).Implements f) eq).to_iff

theorem logical_implements (s : Shape) (c : Config) (rest : Algebra ℝ)
    (order : List Program) (schedule : order.Perm (programs s c)) :
    logicalIO s c rest order ⊨ fun xs batch row col =>
      ∑ k ∈ Finset.range s.k, xs.aAt batch.val row.val k * xs.bAt batch.val k col.val :=
  (logical_iff s c rest order).mpr (real_complete s c rest order schedule)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Interface
