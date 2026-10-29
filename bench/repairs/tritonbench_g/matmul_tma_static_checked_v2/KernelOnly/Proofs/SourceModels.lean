import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.SourceNative
import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.SourceExecution
import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Real
import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Native
namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourceModels
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited StaticChecked.Native.Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The historical typed source has a different AST and inherited dot context.
The real interpretation erases conversions and does not use that context. -/
theorem real_execution (rest : Algebra ℝ) (p : Parameters) (half : Bool) (s : State ℝ) :
    exec (backend (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot) (source p half) s =
    exec (backend (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot) (FPValue.source p half) s := by
  cases half <;> simp [source,FPValue.source,typed,outputDType,FP.Structural.exec,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,SourceExecution.overwriteReg,store,Option.bind,numeric,bop,Region.cast,
    backend,StaticChecked.RealModel.algebra,StaticChecked.RealModel.dot,ofFloat,toFloat,
    ComputeDType.eraseDType]

/-- The selected fixed-instruction dot ignores the inherited computation
context; only its proved same-format conversion identities are used here. -/
theorem native_execution (rest : Algebra StaticChecked.Native.Word)
    (I : StaticChecked.Native.Instructions) (profile : StaticChecked.Native.Profile)
    (p : Parameters) (half : Bool) (s : State StaticChecked.Native.Word) :
    exec (backend (SourceNative.algebra rest I) (StaticChecked.Native.sourceDot I profile)) (source p half) s =
    exec (backend (StaticChecked.Native.algebra rest I) (StaticChecked.Native.sourceDot I profile)) (FPValue.source p half) s := by
  cases half <;> simp [source,FPValue.source,typed,outputDType,FP.Structural.exec,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,SourceExecution.overwriteReg,store,Option.bind,numeric,bop,Region.cast,
    backend,SourceNative.algebra,StaticChecked.Native.algebra,StaticChecked.Native.sourceDot,ofFloat,toFloat,
    ComputeDType.eraseDType]

#axiomsClean real_execution
#axiomsClean native_execution
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourceModels
