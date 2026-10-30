import bench.tritonbench_g.attn_fwd_triton.Proofs.FirstSetup

/-! Bind both checked helper setups to the complete candidate source. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SetupSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem first (p : Parameters) (BM BN BD : Nat) :
    firstSetup BM BN BD p = FirstSetup.code BM BN BD (decide (p.stage=2)) := by
  simp only [firstSetup, Verified.code, Verified.source, TypedSource.repaired,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp [FirstSetup.code,FirstSetup.copies,RealFirstSetup.copies,Setup.code,Setup.upperExpr,
    Setup.lowerExpr,Setup.lowerName,Setup.upperName,Setup.keyBase,Setup.scaleBase,Setup.valueBase,
    RealSetup.lowerExpr,RealSetup.upperExpr,RealSetup.lowerName,RealSetup.upperName,
    RealSetup.keyBase,RealSetup.scaleBase,RealSetup.valueBase,keyName,scaleName,valueName,
    RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName]

theorem second (p : Parameters) (BM BN BD : Nat) :
    secondSetup BM BN BD p = Setup.code BM BN BD Bool.true := by
  simp only [secondSetup, Verified.code, Verified.source, TypedSource.repaired,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp [Setup.code,Setup.upperExpr,Setup.lowerExpr,Setup.lowerName,Setup.upperName,
    Setup.keyBase,Setup.scaleBase,Setup.valueBase,
    RealSetup.lowerExpr,RealSetup.upperExpr,RealSetup.lowerName,RealSetup.upperName,
    RealSetup.keyBase,RealSetup.scaleBase,RealSetup.valueBase,keyName,scaleName,valueName,
    RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SetupSource
