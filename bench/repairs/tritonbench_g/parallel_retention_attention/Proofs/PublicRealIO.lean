import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicRealModel
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicForwardIO
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicBackwardIO

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section

theorem forward_formula_real (p : DotTrees) (s : Shape) (tile : Tile) :
    forwardFormula realScalar p s tile=forwardMath s tile := by
  funext xs o i
  fin_cases o
  change forwardLane realOps (plans s p) s (forwardLogical 0 s xs) tile i.1.val i.2.1.val = _
  exact forward_causal (plans s p) s _ tile i.1.val i.2.1.val i.1.isLt

theorem backward_formula_real (p : DotTrees) (s : Shape) (tile : Tile) :
    backwardFormula realScalar p s tile=backwardMath s tile := by
  funext xs o i
  fin_cases o
  · change queryLane realOps (plans s p) s (backwardLogical 0 s xs) tile i.1.val i.2.1.val = _
    exact query_causal (plans s p) s _ tile i.1.val i.2.1.val i.1.isLt
  · change (keyValueLane realOps (plans s p) s (backwardLogical 0 s xs) tile i.1.val i.2.1.val 0).1 = _
    rw [keyValue_causal (plans s p) s _ tile i.1.val i.2.1.val 0 i.1.isLt]
    rfl
  · change (keyValueLane realOps (plans s p) s (backwardLogical 0 s xs) tile i.1.val 0 i.2.1.val).2 = _
    rw [keyValue_causal (plans s p) s _ tile i.1.val 0 i.2.1.val i.1.isLt]
    rfl

theorem forward_real (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) :
    (forwardIO realScalar p s tile).Implements (forwardMath s tile) := by
  rw [← forward_formula_real p s tile]
  exact forward_implements realScalar p s tile valid

theorem backward_real (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (keyPositive : 0<s.BK) (valuePositive : 0<s.BV) :
    (backwardIO realScalar p s tile).Implements (backwardMath s tile) := by
  rw [← backward_formula_real p s tile]
  exact backward_implements realScalar p s tile valid keyPositive valuePositive

#axiomsClean forward_real
#axiomsClean backward_real
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
