import bench.tritonbench_g.softmax_reducev.Proofs.FPSource
import bench.tritonbench_g.softmax_reducev.SoftmaxReducev
namespace VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) : ComputeKernel :=
  softmax_reducev_surface p.logics p.values p.output p.locations p.starts p.lengths p.maxLength
    p.logicHeadStride p.logicSequenceStride p.valueBatchStride p.valueHeadStride p.valueDimStride
    p.outputBatchStride p.outputHeadStride p.outputDimStride p.locationBatchStride p.locationSequenceStride
    p.dimensions p.tokens p.sentinel

/-- Every original control statement, stride, gather address, mask and loop
bound is preserved. This erasure bridge is a source-address check only;
separate value theorems execute typed cells and explicit fp32 operations. -/
theorem source_projection (p : Parameters) :
    (source p).eraseDType.surfaceBody = (legacySource p).eraseDType.surfaceBody := by
  simp [source,legacySource,typed,softmax_reducev_surface,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody,ComputeDType.eraseDType,Except.bind,bind,pure,
    VeriTile.Triton.eraseDType,FloatDType.numericDType,NumericDType.eraseDType]
  all_goals repeat' constructor
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals try simp [Op.eraseDType,VeriTile.Triton.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals try simp only [NumericDType.eraseDType,eq_mp_eq_cast,eq_mpr_eq_cast,cast_eq]
  all_goals repeat' constructor
  all_goals rfl

/-- Actual default maximum, with its original reduction-first operand order. -/
theorem parsed_maximum (p : Parameters) :
    (loopBody p)[3]? = some (.assign .real [] "n_e_max" (.compute (.alg .fp32
      (.maximumFloat .real .nil
        (.reduceMax (shape := [p.tokens]) ⟨0,by simp⟩ Bool.false (.ref .real [p.tokens] "qk"))
        (.ref .real [] "e_max"))))) := rfl

/-- The signed gather and all ten statements stay inside the full raw loop. -/
theorem parsed_loop (p : Parameters) :
    (source p).surfaceBody.length = 17 ∧ (loopBody p).length = 10 ∧
    (loopBody p)[0]? = some (.assign .nat [] "start_n" (.alg (.ref .nat [] "start_n"))) := by
  constructor <;> first | rfl | exact ⟨rfl,rfl⟩
#axiomsClean source_projection
#axiomsClean parsed_maximum
#axiomsClean parsed_loop
end
end VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPValue
