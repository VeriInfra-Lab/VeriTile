import bench.tritonbench_g.block_sparse_attn.Compatibility
import bench.tritonbench_g.block_sparse_attn.Proofs.RealBackend

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

abbrev R := BlockSparseAttn.Real.R
def mathArithmetic : Arithmetic R := Arithmetic.ofLegacy BlockSparseAttn.Real.ops (fun x => some x) (fun x => some x)

theorem math_add_zero (x : R) : mathArithmetic.add x mathArithmetic.zero=x := by
  cases x <;> simp [mathArithmetic,Arithmetic.ofLegacy,BlockSparseAttn.Real.ops,WithBot.realAdd]
theorem math_mul_comm (x y : R) : mathArithmetic.mul x y=mathArithmetic.mul y x := by
  cases x <;> cases y <;>
    simp [mathArithmetic,Arithmetic.ofLegacy,BlockSparseAttn.Real.ops,WithBot.realMul,mul_comm]
theorem math_fma (x y z : R) :
    mathArithmetic.fma x y z=mathArithmetic.add (mathArithmetic.mul x y) z := rfl
theorem math_mma_add (left right : Fin 8 → R) (x y : R) :
    mathArithmetic.mma8 .mma16n8k8 left right (mathArithmetic.add x y) =
      mathArithmetic.add x (mathArithmetic.mma8 .mma16n8k8 left right y) := by
  cases x <;> cases y <;>
    simp [mathArithmetic,Arithmetic.ofLegacy,BlockSparseAttn.Real.ops,
      FlashAttn.Real.mma8,WithBot.realAdd,WithBot.map,add_assoc]

/-- Exact-real-only seed transport. This theorem is not asserted for native
words or split-rounded floating arithmetic. -/
theorem math_chain_add {k : Nat} (left right : Fin k → Replicas R) (seed : R) :
    chain mathArithmetic left right seed =
      mathArithmetic.add seed (chain mathArithmetic left right mathArithmetic.zero) := by
  let step := fun old chunk => mathArithmetic.mma8 .mma16n8k8
    (fragment mathArithmetic left chunk) (fragment mathArithmetic right chunk) old
  have aux (ns : List Nat) (x y : R) :
      ns.foldl step (mathArithmetic.add x y)=mathArithmetic.add x (ns.foldl step y) := by
    induction ns generalizing y with
    | nil => rfl
    | cons n ns ih =>
      simp only [List.foldl_cons,step,math_mma_add]
      exact ih _
  simpa only [math_add_zero] using aux (List.range (k/8)) seed mathArithmetic.zero

#axiomsClean math_add_zero
#axiomsClean math_mul_comm
#axiomsClean math_fma
#axiomsClean math_mma_add
#axiomsClean math_chain_add
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
