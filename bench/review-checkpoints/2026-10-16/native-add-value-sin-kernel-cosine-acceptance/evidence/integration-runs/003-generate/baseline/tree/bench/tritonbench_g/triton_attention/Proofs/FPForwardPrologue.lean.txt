import bench.tritonbench_g.triton_attention.Proofs.FPForwardInvariant
namespace VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def forwardPrologueResult (P : Primitives α) (p : ForwardParameters) (s : State α) : State α :=
  let t := s.setReg "start_m" .nat [] (fun _ => s.pids 0)
  let t := t.setReg "off_hz" .nat [] (fun _ => s.pids 1)
  let t := t.setReg "offs_m" .nat [p.blockM] (fun i => forwardQueryRow p s i.1)
  let t := t.setReg "offs_n" .nat [p.blockN] (fun i => i.1.val)
  let t := t.setReg "m_prev" .real [p.blockM] (forwardInitial P p).maximum
  let t := t.setReg "l_prev" .real [p.blockM] (forwardInitial P p).denominator
  let t := t.setReg "acc" .real [p.blockM,p.head] (forwardInitial P p).numerator
  let t := t.setReg "stride_qh_2d" .nat [] (fun _ => p.headStride/p.head)
  let t := t.setReg "q_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => forwardQPointer p s)
  let t := t.setReg "k_tile_ptr" .blockPtr [p.blockN,p.head] (fun _ => forwardKVPointer p s p.k 0)
  let t := t.setReg "v_tile_ptr" .blockPtr [p.blockN,p.head] (fun _ => forwardKVPointer p s p.v 0)
  let t := t.setReg "out_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => forwardOutPointer p s)
  t.setReg "q" p.dtype.toTileDType [p.blockM,p.head] (forwardQ p s)

theorem forwardPrologueExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters) (s : State α) :
    run (backend rest P) (forwardPrologue p) s=some (forwardPrologueResult P p s) := by
  simp [forwardPrologue,forwardSource,forwardTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,evalOffsets,backend,forwardPrologueResult,
    forwardInitial,forwardZero,forwardQPointer,forwardKVPointer,forwardOutPointer,
    BlockPtr.inBounds,TileShape.indexToList,forwardQ,forwardOffset,forwardQueryRow,forwardHeadRow,
    Region.cast,FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,Nat.add_assoc]
  delta FP.Structural.bop forwardQ forwardOffset forwardQueryRow forwardHeadRow
  simp only [integralFloorDiv,Nat.div_one,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]

theorem forwardPrologueInvariant (P : Primitives α) (p : ForwardParameters) (s : State α) :
    ForwardInvariant P p s 0 (forwardInitial P p) (forwardPrologueResult P p s) := by
  constructor <;> simp [forwardPrologueResult]

#axiomsClean forwardPrologueExecution
#axiomsClean forwardPrologueInvariant
end
end VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
