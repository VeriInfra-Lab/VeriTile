import bench.tritonbench_g.triton_attention.Proofs.FPBackwardOuterStart
namespace VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def backwardScoreResult (P : Primitives α) (p : BackwardParameters) (initial current : State α)
    (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p) : State α :=
  let key := backwardRead p initial outerMem p.k c
  let t := current.setReg "start_m" .nat [] (fun _ => j*p.blockM)
  let t := t.setReg "offs_m_curr" .nat [p.blockM] (fun i => j*p.blockM+i.1.val)
  let t := t.setReg "q" p.dtype.toTileDType [p.blockM,p.head] (backwardRead p initial st.mem p.q j)
  let t := t.setReg "qk" .fp32 [p.blockM,p.blockM] (backwardQK P p initial st.mem key j)
  let t := t.setReg "qk" .real [p.blockM,p.blockM] (backwardMaskedQK P p initial st.mem key c j)
  let t := t.setReg "m" .fp32 [p.blockM] (fun i => (st.mem p.maximum (backwardRowOffset p initial j i)).read .fp32)
  t.setReg "p" .real [p.blockM,p.blockM] (backwardWeights P p initial st.mem key c j)

theorem backwardScoreExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p)
    (h : BackwardInnerInvariant p initial c j outerMem st current) :
    run (backend rest P) ((backwardInnerBody p).take 6) (current.setReg "start_m" .nat [] (fun _ => j*p.blockM))=
      some (backwardScoreResult P p initial current c j outerMem st) := by
  simp [backwardInnerBody,backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,numeric,numericLe,natLe,backend,backwardScoreResult,
    h.rows,h.cols,h.key,h.qPtr,h.maxPtr,h.memory,h.undef,
    backwardPointer,backwardRead,backwardOffset,backwardRowOffset,
    TileShape.indexToList,Region.cast,Nat.add_assoc,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta backwardQK backwardMaskedQK backwardWeights backwardRead backwardOffset backwardRowOffset FP.Structural.bop
  simp [backwardPointer,TileShape.indexToList,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc,transposeValues]
  all_goals rfl

theorem backwardScoreInvariant (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p)
    (h : BackwardInnerInvariant p initial c j outerMem st current) :
    BackwardInnerInvariant p initial c j outerMem st
      (backwardScoreResult P p initial current c j outerMem st) := by
  constructor
  · constructor
    · constructor <;> simp [backwardScoreResult,h.undef,h.pids,h.hz,h.z,h.h,h.strideZ,h.strideH]
    all_goals simp [backwardScoreResult,h.memory,h.keyPtr,h.valuePtr,h.dkPtr,h.dvPtr]
  all_goals simp [backwardScoreResult,h.lo,h.rows,h.cols,h.qPtr,h.doPtr,h.dqPtr,h.deltaPtr,h.maxPtr,h.key,h.value,h.dqDeadPtrs,h.dk,h.dv]

#axiomsClean backwardScoreExecution
#axiomsClean backwardScoreInvariant
end
end VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
