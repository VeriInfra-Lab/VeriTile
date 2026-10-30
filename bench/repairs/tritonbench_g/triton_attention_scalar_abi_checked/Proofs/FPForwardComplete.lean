import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPForwardBody
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPPreprocess
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
theorem forwardLoopExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial current : State α) (c : Nat) (hc : c≤forwardBlocks p initial) (hn : 0<p.blockN)
    (hd : p.blockN ∣ forwardStop p initial)
    (h : ForwardInvariant P p initial c (forwardRounds P p initial c) current) :
    ∃ final, FP.Structural.range (backend rest P) "start_n" (c*p.blockN)
      (forwardStop p initial) p.blockN (forwardBody p) current=some final ∧
      ForwardInvariant P p initial (forwardBlocks p initial)
        (forwardRounds P p initial (forwardBlocks p initial)) final := by
  have hstop : forwardStop p initial=forwardBlocks p initial*p.blockN := (Nat.div_mul_cancel hd).symm
  have aux : ∀ fuel c (current : State α), forwardBlocks p initial-c=fuel → c≤forwardBlocks p initial →
      ForwardInvariant P p initial c (forwardRounds P p initial c) current →
      ∃ final, FP.Structural.range (backend rest P) "start_n" (c*p.blockN)
        (forwardStop p initial) p.blockN (forwardBody p) current=some final ∧
        ForwardInvariant P p initial (forwardBlocks p initial)
          (forwardRounds P p initial (forwardBlocks p initial)) final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro c current he hc h
      by_cases hb : c<forwardBlocks p initial
      · let next := forwardBodyResult P p initial current c (forwardRounds P p initial c)
        have hi : ForwardInvariant P p initial (c+1) (forwardRounds P p initial (c+1)) next :=
          forwardBodyInvariant P p initial current c _ h
        obtain ⟨final,hr,hf⟩ := ih (forwardBlocks p initial-(c+1)) (by omega) (c+1) next rfl (by omega) hi
        refine ⟨final,?_,hf⟩
        rw [FP.Structural.range,if_neg (by omega),
          if_pos (by rw [hstop]; exact Nat.mul_lt_mul_of_pos_right hb hn),
          forwardBodyExecution rest P p initial current c _ hn h]
        simpa only [Nat.add_mul,Nat.one_mul] using hr
      · have hb : c=forwardBlocks p initial := by omega
        subst c
        refine ⟨current,?_,h⟩
        rw [FP.Structural.range,if_neg (by omega),if_neg (by rw [hstop]; exact Nat.lt_irrefl _)]
  exact aux _ c current rfl hc h

def forwardEpilogueRegisters (p : ForwardParameters) (initial current : State α) : State α :=
  let t := current.setReg "start_m" .nat [] (fun _ => initial.pids 0)
  let t := t.setReg "offs_m" .nat [p.blockM] (fun i => forwardQueryRow p initial i.1)
  let t := t.setReg "l_ptrs" .ptr [p.blockM] (fun i => (p.denominator,forwardRowOffset p initial i))
  t.setReg "m_ptrs" .ptr [p.blockM] (fun i => (p.maximum,forwardRowOffset p initial i))
def forwardEpilogueResult (P : Primitives α) (p : ForwardParameters) (initial current : State α) : State α :=
  let t := (TileShape.allIndices [p.blockM]).foldl (fun t i =>
    t.write p.denominator (forwardRowOffset p initial i) (forwardLCell P p initial i)) (forwardEpilogueRegisters p initial current)
  let t := (TileShape.allIndices [p.blockM]).foldl (fun t i =>
    t.write p.maximum (forwardRowOffset p initial i) (forwardMCell P p initial i)) t
  let t := t.setReg "acc" .fp16 [p.blockM,p.head] (fun i =>
    P.cast none .real .fp16 ((forwardRounds P p initial (forwardBlocks p initial)).numerator i))
  (TileShape.allIndices [p.blockM,p.head]).foldl (fun t i =>
    if (forwardOutPointer p initial).inBounds (TileShape.indexToList _ i) [0,1] then
      t.write p.output (forwardOffset p initial i) (forwardOutput P p initial i) else t) t

theorem forwardEpilogueExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial current : State α)
    (h : ForwardInvariant P p initial (forwardBlocks p initial)
      (forwardRounds P p initial (forwardBlocks p initial)) current) :
    run (backend rest P) (forwardEpilogue p) current=some (forwardEpilogueResult P p initial current) := by
  simp [forwardEpilogue,forwardSource,forwardTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,store,backend,h.start,h.pids,h.hz,h.numerator,h.denominator,h.maximum,h.outputPtr,
    forwardEpilogueResult,forwardEpilogueRegisters,forwardLCell,forwardMCell,forwardOutput,
    forwardRowOffset,forwardQueryRow,forwardOutPointer,forwardQPointer,
    Region.cast,FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,TileShape.indexToList]
  delta forwardOffset forwardQueryRow forwardHeadRow FP.Structural.bop
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,BlockPtr.address_2d_offsets,
    Nat.add_zero,Nat.mul_one,Nat.zero_add,Nat.add_assoc]

theorem maskedScatterMemory {ι : Type} (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (mask : ι → Bool) (l : List ι) (initial current : State α) (h : current.mem=initial.mem) :
    (l.foldl (fun t i => if mask i then t.write r (off i) (v i) else t) current).mem =
      (l.foldl (fun t i => if mask i then t.write r (off i) (v i) else t) initial).mem := by
  induction l generalizing current initial with
  | nil => exact h
  | cons i l ih =>
    simp only [List.foldl_cons]
    apply ih
    split <;> simp_all [State.write]

theorem forwardFinalMemory (P : Primitives α) (p : ForwardParameters) (initial current : State α)
    (h : current.mem=initial.mem) :
    (forwardEpilogueResult P p initial current).mem=(forwardWrite P p initial initial).mem := by
  apply maskedScatterMemory
  simp only [State.setReg_mem]
  apply scatterMemory
  apply scatterMemory
  simpa only [forwardEpilogueRegisters,State.setReg_mem] using h

theorem forwardFullExecution (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial : State α) (hn : 0<p.blockN) (hd : p.blockN ∣ forwardStop p initial) :
    ∃ final, FP.Structural.exec (backend rest P) (forwardSource p) initial=some final ∧
      final.mem=(forwardWrite P p initial initial).mem := by
  let start := forwardPrologueResult P p initial
  obtain ⟨middle,hm,hi⟩ := forwardLoopExecution rest P p initial start 0 (by omega) hn hd
    (forwardPrologueInvariant P p initial)
  refine ⟨forwardEpilogueResult P p initial middle,?_,forwardFinalMemory P p initial middle hi.memory⟩
  change run (backend rest P) (forwardSource p).surfaceBody initial=_
  rw [forwardDecomposition]
  have hstart := forwardPrologueExecution rest P p initial
  have hloop : run (backend rest P)
      [.forRangeDyn "start_n" (.constNat 0)
        (.mul .nat .nil (.add .nat .nil (.ref .nat [] "start_m") (.constNat 1)) (.constNat p.blockM))
        (.constNat p.blockN) (forwardBody p)] start=some middle := by
    have hs := forwardPrologueInvariant P p initial
    simpa [run,step,FP.Structural.evalOp,numeric,hs.start,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,forwardStop] using hm
  dsimp only [start] at hloop
  simp only [run_append,hstart,Option.bind_some,hloop]
  exact forwardEpilogueExecution rest P p initial middle hi

#axiomsClean forwardLoopExecution
#axiomsClean forwardEpilogueExecution
#axiomsClean forwardFullExecution
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
