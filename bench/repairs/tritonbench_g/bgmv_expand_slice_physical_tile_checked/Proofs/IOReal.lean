import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.IOImplementation
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Targets

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicMath
open VeriTile.Triton FP.Structural Direct Direct.FPValues
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

theorem localValue_eq (h : Call) (x w : Nat → ℝ) (old : ℝ) :
    PublicIO.localValue Real.algebra Real.reduction h x w old=
      (if h.add then (∑ k : Fin h.K,x k.val*w k.val)+old
       else ∑ k : Fin h.K,x k.val*w k.val) := by
  have product (k : Fin h.BK) :
      (if h.even=Bool.true ∨ k.val<h.K then x k.val else 0)*
        (if k.val<h.K then w k.val else 0)=
      if k.val<h.K then x k.val*w k.val else 0 := by
    by_cases hk : k.val<h.K <;> simp [hk]
  have total : (∑ k : Fin h.BK,
      (if h.even=Bool.true ∨ k.val<h.K then x k.val else 0)*
        (if k.val<h.K then w k.val else 0))=∑ k : Fin h.K,x k.val*w k.val := by
    simp_rw [product]
    exact Real.padded_sum (fun k => x k*w k) h.K h.BK h.rankCovered
  simpa [PublicIO.localValue,Real.algebra,Real.reduction,FPScalarTree.inputs,
    ofFloat,toFloat,TileShape.insertAxisIndex,show (0.0 : ℝ)=0 by norm_num] using
    congrArg (fun sum : ℝ => if h.add then sum+old else sum) total

theorem implements (h : Call) :
    (PublicIO.sourceIO Real.algebra Real.reduction id h).ImplementsOn PublicIO.Valid (formula h) := by
  have base := PublicIO.source_implements Real.algebra Real.reduction id h
  simpa only [id_eq,localValue_eq,formula] using base

#axiomsClean implements
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicMath
