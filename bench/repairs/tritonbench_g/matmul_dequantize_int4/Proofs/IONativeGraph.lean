import bench.repairs.tritonbench_g.matmul_dequantize_int4.NativeIO
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IOExecution

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word bounded_integer_cast)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

@[simp] theorem held_initial (kernel : ComputeKernel) (backend : Algebra Word) (s : State Word) :
    (held kernel backend).initial s=s := by
  simp [held,Invocation.initial,bindRegisters]

@[simp] theorem counter_initial (kernel : ComputeKernel) (backend : Algebra Word)
    (k : Nat) (s : State Word) :
    (held kernel backend [counter k]).initial s=s.setReg "k" .nat [] (fun _=>k) := by
  simp [held,Invocation.initial,bindRegisters,counter]

theorem fma_step_eq (actual : Native.Instructions) (p : Parameters) (s : State Word)
    (b : Values Word .int [p.blockInner,p.blockColumns])
    (scales zeros : Values Word .fp16 [p.blockColumns])
    (hb : s.regs .int [p.blockInner,p.blockColumns] "b"=some b)
    (hs : s.regs .fp16 [p.blockColumns] "scales"=some scales)
    (hz : s.regs .fp16 [p.blockColumns] "zeros"=some zeros)
    (bounded : ∀ j,-15≤b j ∧ b j≤15) :
    FP.Structural.step (fmaModel actual) (fmaStmt p) s=Native.fmaAssign actual p s := by
  simp only [fmaStmt,FP.Structural.step,evalExpr,evalOp_unfold,hb,hs,hz,fmaModel,
    FPSoftware.scalar,Int4Matmul.FPSoftware.scalar,ofFloat,toFloat,FloatDType.toTileDType,
    id_eq,bind,Option.bind,
    if_true,or_true,FP.Software.scalarFP32]
  simp only [Native.fmaAssign,hb,hs,hz,bind,Option.bind,pure,Option.pure_def]
  congr 2
  funext j
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,bounded_integer_cast _ (bounded _),
    QuantizeKvTransform.MixedBits.half_embedding]

theorem fma_safe (actual : Native.Instructions) (p : Parameters)
    (bounds : RegionName → Nat) (s : State Word) :
    TraceSafe (fmaModel actual) bounds (fmaKernel p) s := by
  simp [TraceSafe,fmaKernel,fmaStmt,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]
  split <;> trivial

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeGraph
