import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerMax
import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerQuantize
import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerControl
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural ControlArithmetic
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod

def finalScale {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.binary (some .fp32) .real .div (MaxPass.maximum M p s (tripCount p))
    (M.literal (some .fp32) .real 127)

/-- The original scale division and pointer reset between the two passes. -/
def resetExecution {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α)
    (h : MaxPass.Invariant M p original (tripCount p) current) :
    {t : State α // run (backend M integer8) (scaleAndReset p) current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [scaleAndReset,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,numeric,numericLt,
    natLt,bop,backend,ComputeDType.eraseDType,Option.bind,h.offsm,h.offsk,h.maximum,
    ofFloat,toFloat,NumericDType.sub]
  rfl

theorem resetInvariant {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α)
    (h : MaxPass.Invariant M p original (tripCount p) current) :
    QuantizePass.Invariant p original (finalScale M p original) 0
      (resetExecution M integer8 p original current h).val := by
  constructor
  all_goals simp [resetExecution,finalScale,h.pids,h.pidm,h.offsk,h.output,
    QuantizePass.inputOffset,QuantizePass.outputOffset,ofFloat,toFloat,
    show (127.0:ℝ)=127 by norm_num,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try (funext i; simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast])
  all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    ← Int.natCast_emod,← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Nat.zero_add]
  all_goals try rfl

/-- The actual final offsets assignment and implicit FP16 scale store. -/
def tailExecution {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (iterations : Nat)
    (h : QuantizePass.Invariant p original scale iterations current) :
    {t : State α // run (backend M integer8) (epilogue p) current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [epilogue,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,numericLt,
    natLt,bop,backend,ComputeDType.eraseDType,Option.bind,h.pidm,h.scale,
    ofFloat,toFloat,NumericDType.sub]
  rfl

theorem reset_memory {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α)
    (h : MaxPass.Invariant M p original (tripCount p) current) :
    (resetExecution M integer8 p original current h).val.mem=current.mem := by
  simp [resetExecution]

theorem tail_memory {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (iterations : Nat)
    (h : QuantizePass.Invariant p original scale iterations current) :
    (tailExecution M integer8 p original current scale iterations h).val.mem =
      fun region offset => if region=p.scales ∧ offset=original.pids 0*p.scaleStride then
        .mk .fp16 (M.cast none .real .fp16 scale) else current.mem region offset := by
  have one : TileShape.allIndices [1]=[(0,PUnit.unit)] := rfl
  simp [tailExecution,one,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    State.write,ofFloat,toFloat]
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Int.mul_one,Int.add_zero]
  rfl

#axiomsClean resetExecution
#axiomsClean resetInvariant
#axiomsClean tailExecution
#axiomsClean reset_memory
#axiomsClean tail_memory
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
