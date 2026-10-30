import bench.tritonbench_g.fast_rms_layernorm.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightProgram
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000

def source (h : Call) (K : Nat) := (Kernels.weightBackward h K).eraseDType.toAlgKernel
def prefixCode (h : Call) (K : Nat) := (source h K).body.take 3
def loopBody (h : Call) (K : Nat) : List Stmt :=
  match ((source h K).body.drop 3).head? with
  | some (.forRangeDyn _ _ _ _ body) => body
  | _ => []
def tailCode (h : Call) (K : Nat) := (source h K).body.drop 4

theorem source_body (h : Call) (K : Nat) :
    (source h K).body=prefixCode h K++
      [.forRangeDyn "row" (.constNat 0) (.constNat h.rows) (.constNat 1) (loopBody h K)]++tailCode h K := by
  simp [prefixCode,loopBody,tailCode,source,Kernels.weightBackward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind]

#axiomsClean source_body
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightProgram
