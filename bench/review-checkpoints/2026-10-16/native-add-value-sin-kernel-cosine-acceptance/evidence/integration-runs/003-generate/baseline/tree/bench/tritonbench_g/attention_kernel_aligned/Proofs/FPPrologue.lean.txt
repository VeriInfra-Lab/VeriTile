import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPInvariant
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def prologueResult (P : Primitives α) (p : Parameters) (s : State α) : State α :=
  let t := s.setReg "start_m" .nat [] (fun _ => s.pids 0)
  let t := t.setReg "off_hz" .nat [] (fun _ => s.pids 1)
  let t := t.setReg "q_offset" .nat [] (fun _ => headBase p s)
  let t := t.setReg "kv_offset" .nat [] (fun _ => headBase p s)
  let t := t.setReg "Q_block_ptr" .blockPtr [p.blockM,p.head] (fun _ => qPointer p s)
  let t := t.setReg "K_block_ptr" .blockPtr [p.head,p.blockN] (fun _ => kPointer p s 0)
  let t := t.setReg "V_block_ptr" .blockPtr [p.blockN,p.head] (fun _ => vPointer p s 0)
  let t := t.setReg "m_i" .real [p.blockM] (initialAccumulator P p).maximum
  let t := t.setReg "l_i" .real [p.blockM] (initialAccumulator P p).denominator
  let t := t.setReg "acc" .real [p.blockM,p.head] (initialAccumulator P p).numerator
  let t := t.setReg "qk_scale" .real [] (fun _ => scaleValue P p)
  let t := t.setReg "q" .fp16 [p.blockM,p.head] (qValue p s)
  let t := t.setReg "q" .fp16 [p.blockM,p.head] (qScaled P p s)
  let t := t.setReg "lo" .nat [] (fun _ => 0)
  let t := t.setReg "hi" .nat [] (fun _ => p.blockN*p.blocks)
  let t := t.setReg "b_ptr_offsets_m" .nat [p.blockM] (fun i => i.1.val)
  let t := t.setReg "b_offset" .nat [] (fun _ => biasBase p s)
  let t := t.setReg "b_ptr_offsets_n_1" .nat [p.blockN] (fun i => i.1.val%p.biasLast+p.biasLast)
  t.setReg "b1" .fp16 [p.blockM,p.blockN] (b1Value p s)

theorem prologueExecution (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    run (backend rest P) (prologue p) s=some (prologueResult P p s) := by
  simp [prologue,source,typedSurface,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,evalOffsets,backend,prologueResult,
    initialAccumulator,qPointer,kPointer,vPointer,BlockPtr.inBounds,
    TileShape.indexToList,scaleValue,qScaled,qValue,qOffset,b1Value,b1Offset,
    row,headBase,biasBase,Region.cast,FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,Nat.add_assoc]
  delta qValue qScaled scaleValue b1Value b1Offset qOffset row headBase biasBase FP.Structural.bop
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,BlockPtr.address_2d_offsets,
    Nat.add_zero,Nat.mul_one,Nat.zero_add,Nat.add_assoc]
  rfl

theorem prologueInvariant (P : Primitives α) (p : Parameters) (s : State α) :
    Invariant P p s 0 (initialAccumulator P p) (prologueResult P p s) := by
  constructor <;> simp [prologueResult]

#axiomsClean prologueExecution
#axiomsClean prologueInvariant
end
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
