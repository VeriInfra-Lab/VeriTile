import bench.tritonbench_g.triton_matmul.Proofs.FPTypedSource
import bench.tritonbench_g.triton_matmul.TritonMatmul
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) (st : Storage) : ComputeKernel :=
  match st with
  | .half => triton_matmul_f16_surface p.A p.B p.C p.M p.N p.K p.stride_am p.stride_ak p.stride_bk p.stride_bn p.stride_cm p.stride_cn p.BLOCK_SIZE_M p.BLOCK_SIZE_N p.BLOCK_SIZE_K p.GROUP_SIZE_M p.numKBlocks
  | .e4m3 => triton_matmul_f8_surface p.A p.B p.C p.M p.N p.K p.stride_am p.stride_ak p.stride_bk p.stride_bn p.stride_cm p.stride_cn p.BLOCK_SIZE_M p.BLOCK_SIZE_N p.BLOCK_SIZE_K p.GROUP_SIZE_M p.numKBlocks

/-- Both original full-loop surfaces have identical control/address/mask and
three-argument dot projection. Erasure proves no floating-value identity. -/
theorem source_projection (p : Parameters) (st : Storage) :
    (source p st).eraseDType.surfaceBody=(legacySource p st).eraseDType.surfaceBody := by
  cases st <;>
    simp [source,legacySource,typed,dtype,triton_matmul_f16_surface,triton_matmul_f8_surface,
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
end VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
