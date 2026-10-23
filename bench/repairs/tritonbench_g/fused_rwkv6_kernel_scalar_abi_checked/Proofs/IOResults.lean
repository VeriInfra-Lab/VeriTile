import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.IO
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Result
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

theorem real_formula (c : Shape) (scale : ℝ) :
    formula realOps c (sequentialPlans c) scale=Reference.real c scale := by
  funext xs o j
  change Fin 2 at o
  fin_cases o
  · simp only [formula,Reference.real,realOps,scalarPartial,sequentialPlans]
    rw [tree_real _ _ (Tree.chain_legal c.BK),←Fin.sum_univ_eq_sum_range]
    exact Real.partial_scalar c (logicalInputs c xs) scale _ _ _ _ j.2.2.2.1.isLt
  · simp only [formula,Reference.real,realOps]
    exact scalar_real_state c (logicalInputs c xs) _ _ _ c.T j.2.1.isLt j.2.2.1.isLt

theorem real (c : Shape) (hc : Sizes c) (scale : ℝ) :
    (programIO Real.algebra c scale).Implements (Reference.real c scale) := by
  have h := Proof.implements Real.algebra c hc (sequentialPlans c) (Source.real_rows c) scale
  simpa only [Real.source_ops,real_formula] using h


#axiomsClean real
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Result
