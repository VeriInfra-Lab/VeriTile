import bench.pretrain_kernels.tasks.local_grad_norm_foreach_vs_loop.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_grad_norm_foreach_vs_loop
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (_xs : IO.Inputs) : Prop :=
  True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs i0 =>
    let input0 := xs.input0
    let vto_copy : Fin 512 → ℝ := fun i0 => input0 ⟨(0) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy lane)^2)
    let vto_copy_1 : Fin 512 → ℝ := fun i0 => input0 ⟨(1) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_1 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_1 lane)^2)
    let vto_copy_2 : Fin 512 → ℝ := fun i0 => input0 ⟨(2) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_2 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_2 lane)^2)
    let vto_copy_3 : Fin 512 → ℝ := fun i0 => input0 ⟨(3) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_3 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_3 lane)^2)
    let vto_copy_4 : Fin 512 → ℝ := fun i0 => input0 ⟨(4) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_4 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_4 lane)^2)
    let vto_copy_5 : Fin 512 → ℝ := fun i0 => input0 ⟨(5) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_5 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_5 lane)^2)
    let vto_copy_6 : Fin 512 → ℝ := fun i0 => input0 ⟨(6) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_6 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_6 lane)^2)
    let vto_copy_7 : Fin 512 → ℝ := fun i0 => input0 ⟨(7) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_7 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_7 lane)^2)
    let vto_copy_8 : Fin 512 → ℝ := fun i0 => input0 ⟨(8) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_8 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_8 lane)^2)
    let vto_copy_9 : Fin 512 → ℝ := fun i0 => input0 ⟨(9) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_9 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_9 lane)^2)
    let vto_copy_10 : Fin 512 → ℝ := fun i0 => input0 ⟨(10) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_10 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_10 lane)^2)
    let vto_copy_11 : Fin 512 → ℝ := fun i0 => input0 ⟨(11) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_11 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_11 lane)^2)
    let vto_copy_12 : Fin 512 → ℝ := fun i0 => input0 ⟨(12) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_12 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_12 lane)^2)
    let vto_copy_13 : Fin 512 → ℝ := fun i0 => input0 ⟨(13) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_13 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_13 lane)^2)
    let vto_copy_14 : Fin 512 → ℝ := fun i0 => input0 ⟨(14) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_14 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_14 lane)^2)
    let vto_copy_15 : Fin 512 → ℝ := fun i0 => input0 ⟨(15) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_15 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_15 lane)^2)
    let vto_copy_16 : Fin 512 → ℝ := fun i0 => input0 ⟨(16) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_16 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_16 lane)^2)
    let vto_copy_17 : Fin 512 → ℝ := fun i0 => input0 ⟨(17) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_17 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_17 lane)^2)
    let vto_copy_18 : Fin 512 → ℝ := fun i0 => input0 ⟨(18) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_18 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_18 lane)^2)
    let vto_copy_19 : Fin 512 → ℝ := fun i0 => input0 ⟨(19) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_19 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_19 lane)^2)
    let vto_copy_20 : Fin 512 → ℝ := fun i0 => input0 ⟨(20) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_20 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_20 lane)^2)
    let vto_copy_21 : Fin 512 → ℝ := fun i0 => input0 ⟨(21) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_21 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_21 lane)^2)
    let vto_copy_22 : Fin 512 → ℝ := fun i0 => input0 ⟨(22) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_22 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_22 lane)^2)
    let vto_copy_23 : Fin 512 → ℝ := fun i0 => input0 ⟨(23) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_23 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_23 lane)^2)
    let vto_copy_24 : Fin 512 → ℝ := fun i0 => input0 ⟨(24) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_24 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_24 lane)^2)
    let vto_copy_25 : Fin 512 → ℝ := fun i0 => input0 ⟨(25) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_25 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_25 lane)^2)
    let vto_copy_26 : Fin 512 → ℝ := fun i0 => input0 ⟨(26) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_26 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_26 lane)^2)
    let vto_copy_27 : Fin 512 → ℝ := fun i0 => input0 ⟨(27) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_27 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_27 lane)^2)
    let vto_copy_28 : Fin 512 → ℝ := fun i0 => input0 ⟨(28) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_28 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_28 lane)^2)
    let vto_copy_29 : Fin 512 → ℝ := fun i0 => input0 ⟨(29) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_29 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_29 lane)^2)
    let vto_copy_30 : Fin 512 → ℝ := fun i0 => input0 ⟨(30) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_30 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_30 lane)^2)
    let vto_copy_31 : Fin 512 → ℝ := fun i0 => input0 ⟨(31) % 32, Nat.mod_lt _ (by decide)⟩ i0
    let vlinalg_vector_norm_31 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 512, (vto_copy_31 lane)^2)
    let vstack : Fin 32 → ℝ := fun i0 => (if i0.val < 1 then vlinalg_vector_norm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 2 then vlinalg_vector_norm_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 3 then vlinalg_vector_norm_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 4 then vlinalg_vector_norm_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 5 then vlinalg_vector_norm_4 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 6 then vlinalg_vector_norm_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 7 then vlinalg_vector_norm_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 8 then vlinalg_vector_norm_7 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 9 then vlinalg_vector_norm_8 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 10 then vlinalg_vector_norm_9 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 11 then vlinalg_vector_norm_10 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 12 then vlinalg_vector_norm_11 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 13 then vlinalg_vector_norm_12 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 14 then vlinalg_vector_norm_13 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 15 then vlinalg_vector_norm_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 16 then vlinalg_vector_norm_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 17 then vlinalg_vector_norm_16 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 18 then vlinalg_vector_norm_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 19 then vlinalg_vector_norm_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 20 then vlinalg_vector_norm_19 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 21 then vlinalg_vector_norm_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 22 then vlinalg_vector_norm_21 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 23 then vlinalg_vector_norm_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 24 then vlinalg_vector_norm_23 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 25 then vlinalg_vector_norm_24 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 26 then vlinalg_vector_norm_25 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 27 then vlinalg_vector_norm_26 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 28 then vlinalg_vector_norm_27 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 29 then vlinalg_vector_norm_28 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 30 then vlinalg_vector_norm_29 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else (if i0.val < 31 then vlinalg_vector_norm_30 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else vlinalg_vector_norm_31 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)))))))))))))))))))))))))))))))
    let vto_copy_32 : Fin 32 → ℝ := fun i0 => vstack i0
    vto_copy_32 i0

end Contract
end
end VeriTile.Bench.PretrainKernels.local_grad_norm_foreach_vs_loop
