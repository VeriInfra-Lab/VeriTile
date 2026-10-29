import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.FPTypedSource
import bench.tritonbench_g.matmul_leakyrelu_fp8.MatmulLeakyreluFp8
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) (st : Storage) : ComputeKernel :=
  matmul_kernel_surface p.A p.B p.C p.M p.N p.K p.stride_am p.stride_ak p.stride_bk p.stride_bn p.stride_cm p.stride_cn p.BLOCK_SIZE_M p.BLOCK_SIZE_N p.BLOCK_SIZE_K p.GROUP_SIZE_M p.ACTIVATION

/-- The original full-loop surface and explicit Python activation have identical control/address/mask and
live-accumulator dot projection. Erasure proves no floating-value identity. -/
theorem source_projection (p : Parameters) (st : Storage) :
    (source p st).eraseDType.surfaceBody=(legacySource p st).eraseDType.surfaceBody := by
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
  all_goals first | rfl | norm_num
/-- Every previously accepted exact-multiple K domain has exactly its original
number of loop trips. The new full cdiv contract also retains genuine K tails. -/
theorem original_trip_count (p : Parameters) (n : Nat) (hBK : 0<BK p) (hK : p.K=BK p*n) :
    trips p=n := by
  unfold trips
  rw [hK]
  have he : BK p*n+BK p-1=(BK p-1)+BK p*n := by omega
  rw [he,Nat.add_mul_div_left _ _ hBK,Nat.div_eq_of_lt (by omega),Nat.zero_add]
#axiomsClean original_trip_count
#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPValue
