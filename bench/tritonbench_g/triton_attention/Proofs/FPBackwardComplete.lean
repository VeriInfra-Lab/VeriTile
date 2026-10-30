import bench.tritonbench_g.triton_attention.Proofs.FPBackwardFinish
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]
theorem backwardOuterExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c : Nat) (mem : Memory α)
    (hc : c≤p.blocks) (hBM : 0<p.blockM) (h : BackwardEntryInvariant p initial c mem current) :
    ∃ final, run (backend rest P) (backwardOuterBody p)
        (current.setReg "start_n" .nat [] (fun _ => c))=some final ∧
      BackwardEntryInvariant p initial (c+1) (backwardOuterAdvance P p initial c mem) final := by
  let start := backwardOuterStartResult P p initial current c mem
  have hi := backwardOuterStartInvariant P p initial current c mem h
  obtain ⟨middle,hm,hmid⟩ := backwardInnerExecution rest P p initial start c 0 mem (by omega) hBM hi
  let st := backwardInnerRounds P p initial mem c (p.blocks-c)
  let final := backwardFinishResult P p initial middle c st
  refine ⟨final,?_,backwardFinishInvariant P p initial middle c mem hmid.toBackwardOuterInvariant⟩
  rw [backwardInnerDecomposition]
  have hs := backwardOuterStartExecution rest P p initial current c mem h
  have hloop : run (backend rest P)
      [.forRangeDyn "start_m" (.ref .nat [] "lo")
        (.mul .nat .nil (.constNat p.blocks) (.constNat p.blockM))
        (.constNat p.blockM) (backwardInnerBody p)] start=some middle := by
    simpa [run,step,FP.Structural.evalOp,numeric,hi.lo,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex] using hm
  dsimp only [start] at hloop
  simp only [run_append,hs,Option.bind_some,hloop]
  exact backwardFinishExecution rest P p initial middle c st hmid.toBackwardOuterInvariant hmid.dv hmid.dk hmid.lo hmid.qPtr hmid.doPtr hmid.dqPtr

theorem backwardOuterLoopExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (c : Nat) (hc : c≤p.blocks) (hBM : 0<p.blockM)
    (h : BackwardEntryInvariant p initial c (backwardRounds P p initial c) current) :
    ∃ final, FP.Structural.range (backend rest P) "start_n" c p.blocks 1 (backwardOuterBody p) current=some final ∧
      BackwardEntryInvariant p initial p.blocks (backwardFinalMemory P p initial) final := by
  have aux : ∀ fuel c (current : State α), p.blocks-c=fuel → c≤p.blocks →
      BackwardEntryInvariant p initial c (backwardRounds P p initial c) current →
      ∃ final, FP.Structural.range (backend rest P) "start_n" c p.blocks 1 (backwardOuterBody p) current=some final ∧
        BackwardEntryInvariant p initial p.blocks (backwardFinalMemory P p initial) final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro c current he hc h
      by_cases hb : c<p.blocks
      · obtain ⟨next,hn,hi⟩ := backwardOuterExecution rest P p initial current c _ hc hBM h
        obtain ⟨final,hr,hf⟩ := ih (p.blocks-(c+1)) (by omega) (c+1) next rfl (by omega) hi
        refine ⟨final,?_,hf⟩
        rw [FP.Structural.range,if_neg (by decide),if_pos hb,hn]
        exact hr
      · have hb : c=p.blocks := by omega
        subst c
        refine ⟨current,?_,h⟩
        rw [FP.Structural.range,if_neg (by decide),if_neg (Nat.lt_irrefl _)]
  exact aux _ c current rfl hc h

theorem backwardFullExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial : State α) (hBM : 0<p.blockM) :
    ∃ final, FP.Structural.exec (backend rest P) (backwardSource p) initial=some final ∧
      final.mem=backwardFinalMemory P p initial := by
  obtain ⟨final,hf,hi⟩ := backwardOuterLoopExecution rest P p initial (backwardPrologueResult p initial) 0
    (by omega) hBM (backwardPrologueInvariant p initial)
  refine ⟨final,?_,hi.memory⟩
  change run (backend rest P) (backwardSource p).surfaceBody initial=_
  rw [backwardDecomposition,run_append,backwardPrologueExecution rest P p initial]
  simp only [Option.bind_some,run,step,hf]
  rfl

#axiomsClean backwardOuterExecution
#axiomsClean backwardOuterLoopExecution
#axiomsClean backwardFullExecution
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
