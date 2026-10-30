import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileReference
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileConfig

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

theorem list_fold_sum {ι : Type} (xs : List ι) (term : ι → Int) (seed : Int) :
    xs.foldl (fun z i => z + term i) seed = seed + (xs.map term).sum := by
  induction xs generalizing seed with
  | nil => simp
  | cons i xs ih => simp [ih, add_assoc]

theorem wordDot_sum {m k n : Nat} (a : TileIndex [m,k] → Int)
    (b : TileIndex [k,n] → Int) (r : Fin m) (c : Fin n) :
    wordDot a b (r,c,PUnit.unit) =
      (BitVec.ofInt 32 ((List.finRange k).map
        (fun q => a (r,q,PUnit.unit) * b (q,c,PUnit.unit))).sum).toInt := by
  unfold wordDot FP.IntegerDot.wordDot
  dsimp only
  change ((List.finRange k).foldl (fun acc q => acc + BitVec.ofInt 32 (a (r,q,PUnit.unit)) *
    BitVec.ofInt 32 (b (q,c,PUnit.unit))) (BitVec.ofInt 32 0)).toInt = _
  rw [FP.IntegerDot.fold_is_low32, list_fold_sum]
  simp

#axiomsClean list_fold_sum
#axiomsClean wordDot_sum
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
