import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelResult
import VeriTile.Triton.Float.IntegerStageSafety

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSafety
open VeriTile.Triton FP.Structural
open KernelSource KernelExecution
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Every actual masked matrix access and the unmasked scale load obey the
public regional bounds. The custom llrint assignment and computed pointer
store use the original integer-stage execution throughout the trace. -/
theorem source_safe {α : Type} [Inhabited α] (algebra : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (version : Bool) (s : State α)
    (bounds : RegionName → Nat)
    (input : ∀ r c, r<p.M → c<p.N → r*p.sam+c*p.san<bounds ⟨p.x⟩)
    (inverse : 0<bounds ⟨p.scale⟩)
    (output : ∀ r c, r<p.M → c<p.N → r+c*p.M<bounds ⟨p.output⟩) :
    QuantizeIntegerStage.TraceSafe algebra (some round) bounds (typed p version) s := by
  cases version <;>
    simp (config := { maxSteps := 1000000 }) [QuantizeIntegerStage.TraceSafe,typed,kernel,
      ComputeKernel.surfaceBody,QuantizeIntegerStage.Safety.traceSafeList_unfold,
      QuantizeIntegerStage.Safety.stepSafe_unfold,QuantizeIntegerStage.Safety.safeExpr_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      QuantizeIntegerStage.exec,QuantizeIntegerStage.run,QuantizeIntegerStage.step,
      QuantizeIntegerStage.evalExpr,QuantizeIntegerStage.store,QuantizeIntegerStage.storedInteger,
      QuantizeIntegerStage.byteOutput,QuantizeIntegerStage.outputRegister,
      FP.Structural.evalExpr,FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,FP.Structural.store,
      FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,
      FP.Structural.natLt,FP.Structural.bop,FP.Structural.integralFloorDiv,FP.Structural.integralMod,
      ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,inverse,State.setReg]
  all_goals constructor
  all_goals
    intro i j hr hc
    first
      | simpa only [Nat.mul_comm p.sam] using input _ _ hr hc
      | exact output _ _ hr hc

end
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSafety
