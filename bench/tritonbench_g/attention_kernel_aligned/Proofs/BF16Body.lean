import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16Score
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue (Primitives backend)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def tailResult (P : Primitives α) (p : Parameters) (initial current : State α)
    (c : Nat) (st : Accumulator α p) : State α :=
  let t := current.setReg "m_i_new" .real [p.blockM] (newMax P p initial c st)
  let t := t.setReg "alpha" .real [p.blockM] (alphaValue P p initial c st)
  let t := t.setReg "p" .real [p.blockM,p.blockN] (weights P p initial c st)
  let t := t.setReg "acc" .real [p.blockM,p.head] (fun i =>
    P.binary (some .fp32) .real .mul (st.numerator i) (alphaValue P p initial c st (i.1,PUnit.unit)))
  let t := t.setReg "acc" .real [p.blockM,p.head] (advance P p initial c st).numerator
  let t := t.setReg "l_i" .real [p.blockM] (advance P p initial c st).denominator
  let t := t.setReg "m_i" .real [p.blockM] (newMax P p initial c st)
  let t := t.setReg "K_block_ptr" .blockPtr [p.head,p.blockN] (fun _ => kPointer p initial (c+1))
  t.setReg "V_block_ptr" .blockPtr [p.blockN,p.head] (fun _ => vPointer p initial (c+1))

def tailPart (p : Parameters) := (body p).drop 7

theorem tailExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (c : Nat) (st : Accumulator α p)
    (hBN : 0<p.blockN) (h : Invariant P p initial c st current)
    (hv : current.regs .bf16 [p.blockN,p.head] "v"=some (vValue P p initial c))
    (hqk : current.regs .bf16 [p.blockM,p.blockN] "qk"=some (score P p initial c)) :
    run (backend rest P) (tailPart p) current=some (tailResult P p initial current c st) := by
  have hn : p.blockN≠0 := by omega
  simp [tailPart,body,source,typedSurface,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,backend,tailResult,hv,hqk,h.maximum,h.denominator,h.numerator,
    h.keyPtr,h.valuePtr,kPointer,vPointer,TileShape.axisDim,hn,hBN,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,Nat.add_mul]
  delta score32 newMax alphaValue weights weightedDot advance FP.Structural.bop
  simp only [Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem tailInvariant (P : Primitives α) (p : Parameters) (initial current : State α)
    (c : Nat) (st : Accumulator α p) (h : Invariant P p initial c st current) :
    Invariant P p initial (c+1) (advance P p initial c st) (tailResult P p initial current c st) := by
  constructor <;> simp [tailResult,advance,h.memory,h.start,h.qoffset,h.ooffset,h.boffset,
    h.brows,h.bias,h.query,h.lo,h.hi]

def bodyResult (P : Primitives α) (p : Parameters) (initial current : State α)
    (c : Nat) (st : Accumulator α p) : State α :=
  tailResult P p initial (scoreResult P p initial current c st) c st

theorem bodyExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (c : Nat) (st : Accumulator α p)
    (hBN : 0<p.blockN) (h : Invariant P p initial c st current) :
    run (backend rest P) (body p) (current.setReg "start_n" .nat [] (fun _ => c*p.blockN))=
      some (bodyResult P p initial current c st) := by
  have split : body p=scorePart p++tailPart p := (List.take_append_drop 7 (body p)).symm
  rw [split,run_append,scoreExecution rest P p initial current c st hBN h]
  exact tailExecution rest P p initial _ c st hBN (scoreInvariant P p initial current c st h)
    (score_value P p initial current c st) (score_score P p initial current c st)

theorem bodyInvariant (P : Primitives α) (p : Parameters) (initial current : State α)
    (c : Nat) (st : Accumulator α p) (h : Invariant P p initial c st current) :
    Invariant P p initial (c+1) (advance P p initial c st) (bodyResult P p initial current c st) :=
  tailInvariant P p initial _ c st (scoreInvariant P p initial current c st h)

#axiomsClean tailExecution
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
