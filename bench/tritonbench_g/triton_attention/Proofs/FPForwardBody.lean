import bench.tritonbench_g.triton_attention.Proofs.FPForwardScore
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def forwardSoftmaxResult (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (c : Nat) (st : ForwardAccumulator α p) : State α :=
  let t := current.setReg "m_curr" .real [p.blockM] (forwardMax P p initial c st)
  let t := t.setReg "l_prev" .real [p.blockM] (forwardOldDenominator P p initial c st)
  let t := t.setReg "p" .real [p.blockM,p.blockN] (forwardWeights P p initial c st)
  let t := t.setReg "l_curr" .real [p.blockM] (forwardDenominator P p initial c st)
  t.setReg "l_rcp" .real [p.blockM] (forwardReciprocal P p initial c st)

def forwardNormalizeResult (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (c : Nat) (st : ForwardAccumulator α p) : State α :=
  let t := current.setReg "p" .real [p.blockM,p.blockN] (forwardNormalized P p initial c st)
  let t := t.setReg "acc" .real [p.blockM,p.head] (forwardRescaled P p initial c st)
  t.setReg "p" .fp16 [p.blockM,p.blockN] (forwardHalfWeights P p initial c st)

def forwardAccumulateResult (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (c : Nat) (st : ForwardAccumulator α p) : State α :=
  let t := current.setReg "v" p.dtype.toTileDType [p.blockN,p.head] (forwardKV p initial p.v c)
  let t := t.setReg "acc" .real [p.blockM,p.head] (forwardAdvance P p initial c st).numerator
  let t := t.setReg "l_prev" .real [p.blockM] (forwardDenominator P p initial c st)
  let t := t.setReg "m_prev" .real [p.blockM] (forwardMax P p initial c st)
  let t := t.setReg "k_tile_ptr" .blockPtr [p.blockN,p.head] (fun _ => forwardKVPointer p initial p.k (c+1))
  t.setReg "v_tile_ptr" .blockPtr [p.blockN,p.head] (fun _ => forwardKVPointer p initial p.v (c+1))

def forwardTailResult (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (c : Nat) (st : ForwardAccumulator α p) : State α :=
  forwardAccumulateResult P p initial
    (forwardNormalizeResult P p initial (forwardSoftmaxResult P p initial current c st) c st) c st

theorem forwardSoftmaxExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial current : State α) (c : Nat) (st : ForwardAccumulator α p)
    (hn : 0<p.blockN)
    (hm : current.regs .real [p.blockM] "m_prev"=some st.maximum)
    (hl : current.regs .real [p.blockM] "l_prev"=some st.denominator)
    (hqk : current.regs .real [p.blockM,p.blockN] "qk"=some (forwardScore P p initial c)) :
    run (backend rest P) (((forwardBody p).drop 5).take 5) current=
      some (forwardSoftmaxResult P p initial current c st) := by
  simp [forwardBody,forwardSource,forwardTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,backend,forwardSoftmaxResult,hqk,hm,hl,TileShape.axisDim,hn,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta forwardMax forwardOldDenominator forwardWeights forwardDenominator forwardReciprocal
  try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  norm_num only
  rfl

theorem forwardNormalizeExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial current : State α) (c : Nat) (st : ForwardAccumulator α p)
    (hp : current.regs .real [p.blockM,p.blockN] "p"=some (forwardWeights P p initial c st))
    (hl : current.regs .real [p.blockM] "l_prev"=some (forwardOldDenominator P p initial c st))
    (hr : current.regs .real [p.blockM] "l_rcp"=some (forwardReciprocal P p initial c st))
    (ha : current.regs .real [p.blockM,p.head] "acc"=some st.numerator) :
    run (backend rest P) (((forwardBody p).drop 10).take 3) current=
      some (forwardNormalizeResult P p initial current c st) := by
  simp [forwardBody,forwardSource,forwardTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,backend,forwardNormalizeResult,hp,hl,hr,ha,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta forwardNormalized forwardRescaled forwardHalfWeights
  try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem forwardAccumulateExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial current : State α) (c : Nat) (st : ForwardAccumulator α p)
    (hmem : current.mem=initial.mem) (hu : current.undef=initial.undef)
    (hk : current.regs .blockPtr [p.blockN,p.head] "k_tile_ptr"=some (fun _ => forwardKVPointer p initial p.k c))
    (hv : current.regs .blockPtr [p.blockN,p.head] "v_tile_ptr"=some (fun _ => forwardKVPointer p initial p.v c))
    (hp : current.regs .fp16 [p.blockM,p.blockN] "p"=some (forwardHalfWeights P p initial c st))
    (hl : current.regs .real [p.blockM] "l_curr"=some (forwardDenominator P p initial c st))
    (hm : current.regs .real [p.blockM] "m_curr"=some (forwardMax P p initial c st))
    (ha : current.regs .real [p.blockM,p.head] "acc"=some (forwardRescaled P p initial c st)) :
    run (backend rest P) ((forwardBody p).drop 13) current=
      some (forwardAccumulateResult P p initial current c st) := by
  simp [forwardBody,forwardSource,forwardTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,backend,forwardAccumulateResult,hmem,hu,hk,hv,hp,hl,hm,ha,
    forwardKVPointer,forwardKV,forwardKVOffset,TileShape.indexToList,Nat.add_mul,Nat.add_assoc,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta forwardAdvance forwardValueDot forwardKV forwardKVOffset
  simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,forwardKVPointer,TileShape.indexToList,Nat.add_assoc]
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Nat.add_mul,Nat.add_assoc]
  rfl

theorem forwardTailExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial current : State α) (c : Nat) (st : ForwardAccumulator α p)
    (hn : 0<p.blockN) (h : ForwardInvariant P p initial c st current)
    (hqk : current.regs .real [p.blockM,p.blockN] "qk"=some (forwardScore P p initial c)) :
    run (backend rest P) ((forwardBody p).drop 5) current=
      some (forwardTailResult P p initial current c st) := by
  have hs := forwardSoftmaxExecution rest P p initial current c st hn h.maximum h.denominator hqk
  have hn := forwardNormalizeExecution rest P p initial (forwardSoftmaxResult P p initial current c st) c st
    (by simp [forwardSoftmaxResult]) (by simp [forwardSoftmaxResult])
    (by simp [forwardSoftmaxResult]) (by simp [forwardSoftmaxResult,h.numerator])
  have ha := forwardAccumulateExecution rest P p initial
    (forwardNormalizeResult P p initial (forwardSoftmaxResult P p initial current c st) c st) c st
    (by simp [forwardNormalizeResult,forwardSoftmaxResult,h.memory])
    (by simp [forwardNormalizeResult,forwardSoftmaxResult,h.undef])
    (by simp [forwardNormalizeResult,forwardSoftmaxResult,h.keyPtr])
    (by simp [forwardNormalizeResult,forwardSoftmaxResult,h.valuePtr])
    (by simp [forwardNormalizeResult])
    (by simp [forwardNormalizeResult,forwardSoftmaxResult])
    (by simp [forwardNormalizeResult,forwardSoftmaxResult])
    (by simp [forwardNormalizeResult])
  have hsplit : (forwardBody p).drop 5 = ((forwardBody p).drop 5).take 5 ++
      ((forwardBody p).drop 10).take 3 ++ (forwardBody p).drop 13 := rfl
  rw [hsplit,run_append,run_append,hs]
  simpa only [Option.bind_some,hn,forwardTailResult] using ha

theorem forwardTailInvariant (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (c : Nat) (st : ForwardAccumulator α p) (h : ForwardInvariant P p initial c st current) :
    ForwardInvariant P p initial (c+1) (forwardAdvance P p initial c st)
      (forwardTailResult P p initial current c st) := by
  constructor <;> simp [forwardTailResult,forwardAccumulateResult,forwardNormalizeResult,forwardSoftmaxResult,forwardAdvance,h.memory,h.undef,h.pids,h.start,h.hz,h.rows,h.cols,
    h.query,h.outputPtr]

def forwardBodyResult (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (c : Nat) (st : ForwardAccumulator α p) : State α :=
  forwardTailResult P p initial (forwardScoreResult P p initial current c) c st

theorem forwardBodyExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial current : State α) (c : Nat) (st : ForwardAccumulator α p)
    (hn : 0<p.blockN) (h : ForwardInvariant P p initial c st current) :
    run (backend rest P) (forwardBody p) (current.setReg "start_n" .nat [] (fun _ => c*p.blockN))=
      some (forwardBodyResult P p initial current c st) := by
  rw [← List.take_append_drop 5 (forwardBody p),run_append,
    forwardScoreExecution rest P p initial current c st h]
  exact forwardTailExecution rest P p initial _ c st hn
    (forwardScoreInvariant P p initial current c st h) (by simp [forwardScoreResult])

theorem forwardBodyInvariant (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (c : Nat) (st : ForwardAccumulator α p) (h : ForwardInvariant P p initial c st current) :
    ForwardInvariant P p initial (c+1) (forwardAdvance P p initial c st)
      (forwardBodyResult P p initial current c st) :=
  forwardTailInvariant P p initial _ c st (forwardScoreInvariant P p initial current c st h)

#axiomsClean forwardTailExecution
#axiomsClean forwardTailInvariant
#axiomsClean forwardBodyExecution
#axiomsClean forwardBodyInvariant
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
