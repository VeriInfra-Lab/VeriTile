import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPPrologue
import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPBody
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loopExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (c : Nat) (hc : c≤p.blocks) (hBN : 0<p.blockN)
    (h : Invariant P p initial c (rounds P p initial c) current) :
    ∃ final, FP.Structural.range (backend rest P) "start_n" (c*p.blockN)
      (p.blockN*p.blocks) p.blockN (body p) current=some final ∧
      Invariant P p initial p.blocks (rounds P p initial p.blocks) final := by
  have aux : ∀ fuel c (current : State α), p.blocks-c=fuel → c≤p.blocks →
      Invariant P p initial c (rounds P p initial c) current →
      ∃ final, FP.Structural.range (backend rest P) "start_n" (c*p.blockN)
        (p.blockN*p.blocks) p.blockN (body p) current=some final ∧
        Invariant P p initial p.blocks (rounds P p initial p.blocks) final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro c current he hc h
      by_cases hb : c<p.blocks
      · let next := bodyResult P p initial current c (rounds P p initial c)
        have hi : Invariant P p initial (c+1) (rounds P p initial (c+1)) next :=
          bodyInvariant P p initial current c _ h
        obtain ⟨final,hr,hf⟩ := ih (p.blocks-(c+1)) (by omega) (c+1) next rfl (by omega) hi
        refine ⟨final,?_,hf⟩
        rw [FP.Structural.range,if_neg (by omega),
          if_pos (by simpa only [Nat.mul_comm p.blockN p.blocks] using Nat.mul_lt_mul_of_pos_right hb hBN),
          bodyExecution rest P p initial current c _ hBN h]
        simpa only [Nat.add_mul,Nat.one_mul] using hr
      · have hb : c=p.blocks := by omega
        subst c
        refine ⟨current,?_,h⟩
        rw [FP.Structural.range,if_neg (by omega),if_neg (by rw [Nat.mul_comm]; exact Nat.lt_irrefl _)]
  exact aux _ c current rfl hc h

def normalized (P : Primitives α) (p : Parameters) (s : State α) (i : QLane p) : α :=
  P.binary (some .fp32) .real .div ((rounds P p s p.blocks).numerator i)
    ((rounds P p s p.blocks).denominator (i.1,PUnit.unit))
def epilogueStart (P : Primitives α) (p : Parameters) (initial current : State α) : State α :=
  (current.setReg "acc" .real [p.blockM,p.head] (normalized P p initial)).setReg
    "O_block_ptr" .blockPtr [p.blockM,p.head] (fun _ => outputPointer p initial)
def epilogueResult (P : Primitives α) (p : Parameters) (initial current : State α) : State α :=
  (TileShape.allIndices [p.blockM,p.head]).foldl (fun t i =>
    t.write p.output (qOffset p initial i) (.mk .fp16 (outputValue P p initial i)))
    (epilogueStart P p initial current)

theorem epilogueExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α)
    (h : Invariant P p initial p.blocks (rounds P p initial p.blocks) current) :
    run (backend rest P) (epilogue p) current=some (epilogueResult P p initial current) := by
  simp [epilogue,source,typedSurface,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,evalOffsets,numeric,store,backend,h.start,h.qoffset,h.numerator,h.denominator,
    epilogueResult,epilogueStart,normalized,outputValue,outputPointer,qPointer,
    BlockPtr.inBounds,TileShape.indexToList,Region.cast,
    FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,qOffset,row,headBase,Nat.add_assoc]
  all_goals delta normalized outputValue qOffset row headBase FP.Structural.bop
  all_goals simp only [Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc,Nat.mul_one,Nat.add_zero]
  all_goals try rfl

private theorem scatter_mem_eq {ι : Type} (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (l : List ι) (initial current : State α) (h : current.mem=initial.mem) :
    (l.foldl (fun t i => t.write r (off i) (v i)) current).mem =
      (l.foldl (fun t i => t.write r (off i) (v i)) initial).mem := by
  induction l generalizing current initial with
  | nil => exact h
  | cons i l ih =>
    simp only [List.foldl_cons]
    apply ih
    simp [State.write,h]

theorem output_injective (p : Parameters) (initial : State α) (hHD : 0<p.head) :
    Function.Injective (qOffset p initial) := by
  intro a b he
  have he' : a.1.val*p.head+a.2.1.val=b.1.val*p.head+b.2.1.val := by
    simp only [qOffset,row,Nat.add_mul] at he
    omega
  have hr := congrArg (fun n => n/p.head) he'
  simp [Nat.add_div,hHD,Nat.div_eq_of_lt a.2.1.isLt,Nat.div_eq_of_lt b.2.1.isLt,
    Nat.mod_eq_of_lt a.2.1.isLt,Nat.mod_eq_of_lt b.2.1.isLt,
    Nat.not_le_of_lt a.2.1.isLt,Nat.not_le_of_lt b.2.1.isLt] at hr
  apply Prod.ext
  · exact Fin.ext hr
  · apply Prod.ext
    · apply Fin.ext
      rw [hr] at he'
      omega
    · exact Subsingleton.elim _ _

theorem correctness (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial : State α) (hBN : 0<p.blockN) (hHD : 0<p.head) :
    ∃ final, FP.Structural.exec (backend rest P) (source p) initial=some final ∧
      final.mem=(written P p initial).mem ∧
      (∀ i : QLane p, final.mem p.output (qOffset p initial i)=
        .mk .fp16 (outputValue P p initial i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : QLane p, o≠qOffset p initial i) →
        final.mem r o=initial.mem r o) := by
  let start := prologueResult P p initial
  obtain ⟨middle,hm,hi⟩ := loopExecution rest P p initial start 0 (by omega) hBN
    (prologueInvariant P p initial)
  let finish := epilogueResult P p initial middle
  have hmem : finish.mem=(written P p initial).mem := by
    apply scatter_mem_eq
    simpa only [epilogueStart,State.setReg_mem] using hi.memory
  refine ⟨finish,?_,hmem,?_,?_⟩
  · change run (backend rest P) (source p).surfaceBody initial=some finish
    rw [source_decomposition]
    have hstart : run (backend rest P) (prologue p) initial=some start := prologueExecution rest P p initial
    have hloop : run (backend rest P)
        [.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat p.blockN) (body p)]
        start=some middle := by
      have hs := prologueInvariant P p initial
      simpa [run,step,FP.Structural.evalOp,hs.lo,hs.hi] using hm
    simp only [run_append,hstart,Option.bind_some,hloop]
    exact epilogueExecution rest P p initial middle hi
  · intro i
    rw [hmem]
    exact State.scatter_readback _ _ _ _ _ (output_injective p initial hHD)
  · intro r o hf
    rw [hmem]
    exact State.scatter_frame _ _ _ _ _ _ (hf.imp id (fun h i _ => h i)) _

#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean output_injective
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
