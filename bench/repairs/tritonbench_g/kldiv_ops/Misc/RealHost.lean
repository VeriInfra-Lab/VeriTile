import bench.repairs.tritonbench_g.kldiv_ops.Proofs.RealFormula
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.ForwardValues
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.BackwardValues

namespace VeriTile.Bench.Repairs.KldivOps.RealExecution
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

def libraryNumerics : Host.Numerics ℝ where
  zero := 0
  one := 1
  neg := fun x=> -x
  equal := fun x y=>decide (x=y)
  sum := fun _ xs=>xs.sum
  divide := fun x n=>x/n
  multiply := fun x y=>x*y

theorem range_sum (n : Nat) (f : Nat→ℝ) : ((List.range n).map f).sum=∑ i : Fin n,f i.val := by
  rw [Fin.sum_univ_eq_sum_range]
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ,Finset.sum_range_succ,ih]

theorem row_sum (flag : Bool) (eps : ℝ) (rows cols : Nat) (prediction target : Nat→ℝ) :
    ((List.range rows).map (fun row=>FPGraph.row algebra cols (Host.blockSize cols) flag eps
      (fun col=>prediction (row*cols+col)) (fun col=>target (row*cols+col)))).sum=
      total flag eps rows cols prediction target := by
  simp_rw [row_ideal flag eps cols (Host.blockSize cols) (Host.block_positive cols)]
  exact range_sum rows _

/-- The exact same complete host reduction graph, interpreted in ℝ, equals
an independently defined logical tensor formula for all four reductions.
No byte codec on ℝ or impossible universal floating/Real equality is assumed. -/
theorem returned_forward_ideal (flag : Bool) (eps : ℝ) (red : Host.Reduction)
    (rows cols : Nat) (prediction target : Nat→ℝ) (i : Nat) :
    Host.reducedWord libraryNumerics red rows cols
      (Host.forwardKernelValue algebra red cols flag eps prediction target) i=
      forward flag eps red.code rows cols prediction target i := by
  have totalRows := row_sum flag eps rows cols prediction target
  unfold Host.forwardKernelValue
  cases red <;>
    simp [Host.reducedWord,libraryNumerics,Host.forwardKernelValue,forward,Host.Reduction.code,
      scalar_real,totalRows]

theorem returned_backward_prefix_ideal (flag : Bool) (count : Nat)
    (target initialWork : Nat→ℝ) (i : Nat) (hi : i<count) :
    Host.backwardKernelValue algebra flag count target initialWork i=backward flag (target i) := by
  simp only [Host.backwardKernelValue,hi,if_true]
  exact backward_scalar_ideal flag (target i)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.RealExecution
