import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.LegacySurface
import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelExecution
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource.SourcePreservation
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- The complete grouped program retains every statement before the store.
The new frontend exposes the genuine implicit INT8 store conversion; the old
surface left a runtime marker and delegated that conversion to the executor. -/
theorem before_store (v : Bool) (p : Parameters) :
    (KernelSource.kernel v p).surfaceBody.take 19 =
      (LegacySurface.kernel v p).surfaceBody.take 19 := by
  cases v <;> rfl

/-- Both entire source/candidate executions retain the same success/failure,
register state and memory for arbitrary scalar primitives and initial states.
The only IR difference is the explicit implicit-store narrowing, which the
retained integer executor already performs at the actual output pointer. -/
theorem kernel {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (v : Bool) (p : Parameters) (s : State α) :
    KernelExecution.exec M round p v s =
      QuantizeIntegerStage.exec M (some round)
        ⟨LegacySurface.kernel v p, QuantizeIntegerStage.byteOutput ⟨p.output⟩,
          QuantizeIntegerStage.outputRegister⟩ s := by
  cases v <;>
    simp only [KernelExecution.exec, KernelExecution.typed, KernelSource.kernel,
      quotedOriginal, quotedCommuted, sourceKernel, commutedSourceKernel,
      LegacySurface.kernel, ComputeKernel.surfaceBody, QuantizeIntegerStage.exec,
      QuantizeIntegerStage.run, QuantizeIntegerStage.step, QuantizeIntegerStage.evalExpr,
      FP.Structural.eval_intToSigned]
  all_goals simp (config := { maxSteps := 1000000 }) [KernelExecution.exec, KernelExecution.typed,
      KernelSource.kernel, quotedOriginal, quotedCommuted, sourceKernel, commutedSourceKernel,
      LegacySurface.kernel, ComputeKernel.surfaceBody,
      QuantizeIntegerStage.exec, QuantizeIntegerStage.run, QuantizeIntegerStage.step,
      QuantizeIntegerStage.evalExpr, QuantizeIntegerStage.store,
      QuantizeIntegerStage.storedInteger, QuantizeIntegerStage.byteOutput,
      QuantizeIntegerStage.outputRegister, FP.Structural.evalExpr,
      FP.Structural.evalComputeOp, FP.Structural.eval_intToSigned, FP.Structural.evalOp_unfold,
      FP.Structural.store, FP.Structural.numeric, FP.Structural.numericLt,
      FP.Structural.numericEq, FP.Structural.natLt, FP.Structural.bop,
      FP.Structural.integralFloorDiv, FP.Structural.integralMod,
      ComputeDType.eraseDType, ofFloat, toFloat, Region.cast, Option.bind,
      BitVec.ofInt_toInt]

#axiomsClean before_store
#axiomsClean kernel
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource.SourcePreservation
