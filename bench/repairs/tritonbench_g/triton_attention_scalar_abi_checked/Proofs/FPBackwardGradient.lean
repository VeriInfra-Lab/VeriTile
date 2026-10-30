import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardScore
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def backwardGradientResult (P : Primitives α) (p : BackwardParameters) (initial current : State α)
    (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p) : State α :=
  let key := backwardRead p initial outerMem p.k c
  let value := backwardRead p initial outerMem p.v c
  let t := current.setReg "do_val" p.dtype.toTileDType [p.blockM,p.head] (backwardRead p initial st.mem p.gradient j)
  let t := t.setReg "dv" .real [p.blockM,p.head] (backwardInnerAdvance P p initial key value c j st).dv
  let t := t.setReg "Di" .fp32 [p.blockM] (fun i => (st.mem p.delta (backwardRowOffset p initial j i)).read .fp32)
  let t := t.setReg "dp" .real [p.blockM,p.blockM] (backwardDPInitial P p initial st.mem j)
  let t := t.setReg "dp" .real [p.blockM,p.blockM] (backwardDP P p initial st.mem value j)
  t.setReg "ds" .real [p.blockM,p.blockM] (backwardDS P p initial st.mem key value c j)

theorem backwardGradientExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c j : Nat) (outerMem : Memory α) (st : BackwardInner α p)
    (h : BackwardInnerInvariant p initial c j outerMem st current)
    (hr : current.regs .nat [p.blockM] "offs_m_curr"=some (fun i => j*p.blockM+i.1.val))
    (hp : current.regs .real [p.blockM,p.blockM] "p"=
      some (backwardWeights P p initial st.mem (backwardRead p initial outerMem p.k c) c j)) :
    run (backend rest P) (((backwardInnerBody p).drop 6).take 6) current=
      some (backwardGradientResult P p initial current c j outerMem st) := by
  simp [backwardInnerBody,backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,numeric,backend,backwardGradientResult,
    h.dv,h.value,h.doPtr,h.deltaPtr,h.memory,h.undef,hr,hp,
    backwardPointer,backwardRead,backwardOffset,backwardRowOffset,
    TileShape.indexToList,Region.cast,Nat.add_assoc,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta backwardInnerAdvance backwardValueDot backwardDPInitial backwardDP backwardDPDot backwardDS
    backwardRead backwardOffset backwardRowOffset FP.Structural.bop
  simp [backwardPointer,TileShape.indexToList,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc,transposeValues]
  all_goals rfl

#axiomsClean backwardGradientExecution
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
