import bench.tritonbench_g.kv_cache_filling.Proofs.TypedUInt8

namespace VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedUInt8
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- The load/address/mask prefix is exactly the old accepted K stage. The
final store deliberately differs: fp32-to-uint8 conversion replaces a Real
store, matching the supplied Python cache dtype. -/
theorem key_prefix_projection (p : Parameters) :
    (keySource p).eraseDType.toAlgKernel.body.dropLast =
      (fill_k_cache_tile p.input p.output p.blockOffsets p.sourceIndex p.blockIndex p.kvBlockIndex
        p.strideSs p.strideSh p.strideSd p.strideCn p.strideCb p.strideCh p.strideCd
        p.strideBoff p.heads p.dimension p.height p.width).toAlgKernel.body.dropLast := by
  simp only [keySource, fill_k_cache_tile,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  rfl

theorem value_prefix_projection (p : Parameters) :
    (valueSource p).eraseDType.toAlgKernel.body.dropLast =
      (fill_v_cache_tile p.input p.output p.blockOffsets p.sourceIndex p.blockIndex p.kvBlockIndex
        p.strideSs p.strideSh p.strideSd p.strideCn p.strideCb p.strideCh p.strideCd
        p.strideBoff p.heads p.dimension p.height p.width).toAlgKernel.body.dropLast := by
  simp only [valueSource, fill_v_cache_tile,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  rfl

#axiomsClean key_prefix_projection
#axiomsClean value_prefix_projection
end VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedUInt8
