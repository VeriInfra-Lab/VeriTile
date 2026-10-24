import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16Invariant
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue (Primitives backend)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def scoreResult (P : Primitives α) (p : Parameters) (initial current : State α)
    (c : Nat) (st : Accumulator α p) : State α :=
  let t := current.setReg "start_n" .nat [] (fun _ => c*p.blockN)
  let t := t.setReg "k" .bf16 [p.head,p.blockN] (kValue P p initial c)
  let t := t.setReg "v" .bf16 [p.blockN,p.head] (vValue P p initial c)
  let t := t.setReg "qk" .bf16 [p.blockM,p.blockN] (fun _ => halfZero P)
  let t := t.setReg "qk" .bf16 [p.blockM,p.blockN]
    (fun i => P.binary none .bf16 .add (halfZero P) (qkDot P p initial c i))
  let t := t.setReg "b0" .bf16 [p.blockM,1] (fun i => b0Value P p initial c i.1)
  let t := t.setReg "qk" .bf16 [p.blockM,p.blockN] (rawScore P p initial c)
  t.setReg "qk" .bf16 [p.blockM,p.blockN] (score P p initial c)
def scorePart (p : Parameters) := (body p).take 7

theorem scoreExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (c : Nat) (st : Accumulator α p)
    (hBN : 0<p.blockN) (h : Invariant P p initial c st current) :
    run (backend rest P) (scorePart p) (current.setReg "start_n" .nat [] (fun _ => c*p.blockN))=
      some (scoreResult P p initial current c st) := by
  have hn : p.blockN≠0 := by omega
  simp [scorePart,body,source,typedSurface,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,backend,scoreResult,h.memory,h.start,h.qoffset,h.ooffset,h.boffset,
    h.brows,h.bias,h.query,h.keyPtr,h.valuePtr,kPointer,vPointer,integralMod,natLt,Fin.isLt,FP.Structural.boundaryZero,
    TileShape.indexToList,Region.cast,hn,hBN,FP.Structural.ofFloat,FP.Structural.toFloat,
    ComputeDType.eraseDType,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta score rawScore qkDot halfZero kValue vValue kOffset vOffset b0Value b0Offset headBase kvBase valueBase biasBase row FP.Structural.bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,BlockPtr.address_2d_offsets,integralFloorDiv,hn,hBN,Nat.add_assoc]
  try rfl

theorem scoreInvariant (P : Primitives α) (p : Parameters) (initial current : State α)
    (c : Nat) (st : Accumulator α p) (h : Invariant P p initial c st current) :
    Invariant P p initial c st (scoreResult P p initial current c st) := by
  constructor <;> simp [scoreResult,h.memory,h.start,h.qoffset,h.ooffset,h.boffset,h.brows,h.bias,
    h.query,h.keyPtr,h.valuePtr,h.maximum,h.denominator,h.numerator,h.lo,h.hi]

theorem score_key (P : Primitives α) (p : Parameters) (initial current : State α) (c : Nat) (st : Accumulator α p) :
    (scoreResult P p initial current c st).regs .bf16 [p.head,p.blockN] "k"=some (kValue P p initial c) := by simp [scoreResult]
theorem score_value (P : Primitives α) (p : Parameters) (initial current : State α) (c : Nat) (st : Accumulator α p) :
    (scoreResult P p initial current c st).regs .bf16 [p.blockN,p.head] "v"=some (vValue P p initial c) := by simp [scoreResult]
theorem score_score (P : Primitives α) (p : Parameters) (initial current : State α) (c : Nat) (st : Accumulator α p) :
    (scoreResult P p initial current c st).regs .bf16 [p.blockM,p.blockN] "qk"=some (score P p initial c) := by simp [scoreResult]

#axiomsClean scoreExecution
#axiomsClean scoreInvariant
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
