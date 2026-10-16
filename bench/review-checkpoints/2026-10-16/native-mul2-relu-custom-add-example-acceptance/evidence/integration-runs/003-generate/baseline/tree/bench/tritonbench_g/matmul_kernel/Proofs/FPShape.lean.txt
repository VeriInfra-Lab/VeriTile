import bench.tritonbench_g.matmul_kernel.Proofs.FPTypedSource
import bench.tritonbench_g.matmul_kernel.MatmulKernel
namespace VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) : ComputeKernel :=
  matmul_kernel_surface p.C p.A p.B p.BLOCK_SIZE_M p.BLOCK_SIZE_N p.BLOCK_SIZE_K p.numKBlocks

/-- The complete original full-loop surface have identical control/address/unmasked loads/stores and
three-argument dot projection. Erasure proves no floating-value identity. -/
theorem source_projection (p : Parameters) (st : Storage) :
    (source p st).eraseDType.surfaceBody=(legacySource p).eraseDType.surfaceBody := by
  cases st <;>
    simp [source,legacySource,typed,dtype,matmul_kernel_surface,
      ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
      MemAccess.eraseDType,MaskOpt.eraseDType,ComputeKernel.fromKernelBody,
      ComputeKernel.surfaceBody,ComputeDType.eraseDType,Except.bind,bind,pure,
      FloatDType.toTileDType,FloatDType.numericDType]
  all_goals repeat' constructor
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp only [FloatDType.toTileDType,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,Op.eraseDType.eq_def,NumericDType.eraseDType,
    ComparableDType.eraseDType,VeriTile.Triton.eraseDType]
  all_goals try simp [eq_mp_eq_cast,eq_mpr_eq_cast,cast_eq,id_eq]
  all_goals rfl
#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue
