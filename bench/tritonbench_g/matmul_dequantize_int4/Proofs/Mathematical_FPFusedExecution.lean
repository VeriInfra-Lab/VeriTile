import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Mathematical_FPFusedPrefix
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPFused
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
open FPCompiled (DotAcc)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Explicit task-local compiled instruction. It reads actual current SSA
registers (packed nibble, scale, and separately rounded zero product), performs
half sign-bit negation followed by one-round half FMA, and writes b. There is
no read from a specification result or arbitrary numerical callback. -/
def fmaAssign (p : Parameters) (current : State Word) : Option (State Word) := do
  let b ← current.regs .int [p.blockInner,p.blockColumns] "b"
  let scale ← current.regs .fp16 [p.blockColumns] "scales"
  let zero ← current.regs .fp16 [p.blockColumns] "zeros"
  pure (current.setReg "b" .fp16 [p.blockInner,p.blockColumns] (fun i =>
    QuantizeKvTransform.MixedBits.embed (FPCompiled.HalfFMA.fma .ieeeValue
      (MatmulDequantInt4.Binary16.round .ieeeValue (b i : Rat))
      (QuantizeKvTransform.MixedBits.half (scale (i.2.1,PUnit.unit)))
      (QuantizeKvTransform.MixedBits.half (zero (i.2.1,PUnit.unit))).neg)))

def middleState (M : Algebra Word) (D : DotAcc Word) (p : Parameters)
    (initial current : State Word) (k : Nat) : State Word :=
  current.setReg "b" .fp16 [p.blockInner,p.blockColumns] (bValue M p initial k)

theorem fmaAssign_correct (M : Algebra Word) (D : DotAcc Word) (p : Parameters)
    (initial current : State Word) (k : Nat) (h : PrefixFacts M D p initial k current) :
    fmaAssign p current = some (middleState M D p initial current k) := by
  simp only [fmaAssign,h.b,h.scales,h.zeros,Option.bind_some,pure,Option.pure_def]
  rfl

/-- Actual suffix consists of live-carry dot and the two original pointer
increments. It is interpreted by the shared Structural evaluator. -/
def suffixExecution (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial current : State Word) (k : Nat)
    (h : PrefixFacts M D p initial k current) :
    {t : State Word // run (FPCompiled.backend M D) (suffix p)
      (middleState M D p initial current k) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [suffix,FPCompiled.body,FPCompiled.source,FPCompiled.typed,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
    bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    middleState,h.a,h.control.aptrs,h.control.bptrs,h.control.acc,h.control.inf,FPCompiled.backend]
  rfl

theorem suffixInvariant (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial current : State Word) (k : Nat)
    (h : PrefixFacts M D p initial k current) :
    Invariant M D p initial (k+1) (suffixExecution M D p initial current k h).val := by
  constructor <;>
    simp [suffixExecution,middleState,h.control.memory,h.control.pidm,h.control.pidn,
      h.control.numk,h.control.bits,h.control.inf,h.control.shift,h.control.zshift,
      h.control.amask,h.control.aptrs,h.control.bptrs,h.control.sptrs,h.control.zptrs,
      h.control.scales,h.control.zeros,h.control.acc,h.a,
      FPCompiled.backend,accumulation,aOffset,bOffset,inputInner,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]

  all_goals try exact h.control.scales
  all_goals try exact h.control.zeros
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]

def runBody (M : Algebra Word) (D : DotAcc Word) (p : Parameters)
    (k : Nat) (s : State Word) : Option (State Word) := do
  let start ← run (FPCompiled.backend M D) (loopPrefix p) (s.setReg "k" .nat [] (fun _ => k))
  let mid ← fmaAssign p start
  run (FPCompiled.backend M D) (suffix p) mid

def bodyExecution (M : Algebra Word) (D : DotAcc Word) (p : Parameters)
    (initial current : State Word) (k : Nat) (h : Invariant M D p initial k current) :
    {t : State Word // runBody M D p k current=some t} := by
  let start := prefixExecution M D p initial current k h
  have hs := prefixFacts M D p initial current k h
  let finish := suffixExecution M D p initial start.val k hs
  refine ⟨finish.val,?_⟩
  unfold runBody
  simp only [start.property,Option.bind_some]
  simp only [bind,Option.bind, fmaAssign_correct M D p initial start.val k hs]
  exact finish.property

theorem bodyInvariant (M : Algebra Word) (D : DotAcc Word) (p : Parameters)
    (initial current : State Word) (k : Nat) (h : Invariant M D p initial k current) :
    Invariant M D p initial (k+1) (bodyExecution M D p initial current k h).val :=
  suffixInvariant M D p initial _ k (prefixFacts M D p initial current k h)

#axiomsClean fmaAssign_correct
#axiomsClean suffixExecution
#axiomsClean suffixInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPFused
