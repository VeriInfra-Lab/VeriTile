import bench.tritonbench_g.attn_fwd_triton.Proofs.AttnFwdTriton_Verified_Iteration_Iteration

/-! Full execution of either changed helper range. Bounds and strides remain
symbolic; the closed carry comes from the independent online formula library. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Loop
open VeriTile.Triton Memory Preparation Iteration
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry closed closed_zero closed_succ)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

abbrev cursor := RealLoop.cursor
abbrev count := RealFormula.tripCount

theorem source_of_flat (BM BN BD D : Nat) (second firstDiagonal : Bool) (s t : BlockState)
    (h : stepStmts (flatIteration BM BN BD D second (second || firstDiagonal)) s = some t) :
    stepStmts (sourceIteration BM BN BD D second firstDiagonal) s = some t := by
  exact h

def after (mem : BlockState) (c : Config) (BM BN BD lower : Nat) (second firstDiagonal : Bool)
    (seed : Carry BM BD) (n : Nat) : Carry BM BD :=
  closed seed (fun i => score mem c BM BN BD (cursor lower BN i))
    (fun i => Masks.keep BM BN c.p.context c.startM (cursor lower BN i) (second || firstDiagonal))
    (fun i => vvalue mem c BN BD (cursor lower BN i)) n

structure Registers (mem s : BlockState) (c : Config) (BM BN BD start : Nat)
    (second : Bool) (carry : Carry BM BD) : Prop where
  fixed : Fixed mem s c BM BN BD
  carry : Carries s carry
  pointers : Pointers s c BN BD start second

theorem invariant_step (mem s : BlockState) (c : Config) (BM BN BD lower : Nat)
    (second firstDiagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN) (i : Nat)
    (hr : Registers mem s c BM BN BD (cursor lower BN i) second
      (after mem c BM BN BD lower second firstDiagonal seed i)) :
    ∃ t, stepStmts (sourceIteration BM BN BD c.p.headDim second firstDiagonal)
        (s.setReg "start_n" .int [] (Tile.scalar (cursor lower BN i))) = some t ∧
      Registers mem t c BM BN BD (cursor lower BN (i+1)) second
        (after mem c BM BN BD lower second firstDiagonal seed (i+1)) ∧ Frame (writes second) s t := by
  let si := s.setReg "start_n" .int [] (Tile.scalar (cursor lower BN i))
  have hf0 : Frame (writes second) s si := Frame.setReg _ _ _ _ _ _ (by simp [writes,prefixWrites])
  have hc : Carries si (after mem c BM BN BD lower second firstDiagonal seed i) :=
    ⟨by simpa [si] using hr.carry.center,by simpa [si] using hr.carry.denominator,
      by simpa [si] using hr.carry.numerator⟩
  have hp : Pointers si c BN BD (cursor lower BN i) second := by
    constructor
    · cases second <;> simpa [si,keyName,RealMemory.keyName] using hr.pointers.key
    · cases second <;> simpa [si,scaleName,RealMemory.scaleName] using hr.pointers.scale
    · cases second <;> simpa [si,valueName,RealMemory.valueName] using hr.pointers.value
  obtain ⟨t,ht,hcarry,hptr,hfixed,hf⟩ := Iteration.executes mem si c BM BN BD (cursor lower BN i) hN
    second (second || firstDiagonal) (after mem c BM BN BD lower second firstDiagonal seed i)
    (hr.fixed.transport hf0) hc hp (by simp [si])
  refine ⟨t,source_of_flat BM BN BD c.p.headDim second firstDiagonal si t ht,⟨hfixed,?_,?_⟩,hf0.trans hf⟩
  · simpa only [after,closed_succ,updated] using hcarry
  · simpa only [RealLoop.cursor_succ] using hptr

theorem loop_from (mem s : BlockState) (c : Config) (BM BN BD lower upper : Nat)
    (second firstDiagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (i : Nat) (hi : i ≤ count lower upper BN)
    (hr : Registers mem s c BM BN BD (cursor lower BN i) second
      (after mem c BM BN BD lower second firstDiagonal seed i)) :
    ∃ t, stepForRangeIntAux "start_n" ((cursor lower BN i : Nat) : Int) (upper : Int) BN
        (sourceIteration BM BN BD c.p.headDim second firstDiagonal) s = some t ∧
      Registers mem t c BM BN BD (cursor lower BN (count lower upper BN)) second
        (after mem c BM BN BD lower second firstDiagonal seed (count lower upper BN)) ∧ Frame (writes second) s t := by
  by_cases h : i < count lower upper BN
  · obtain ⟨sm,hs,hnext,hf⟩ := invariant_step mem s c BM BN BD lower second firstDiagonal seed hN i hr
    obtain ⟨t,ht,hfinal,hframe⟩ := loop_from mem sm c BM BN BD lower upper second firstDiagonal seed hN
      (i+1) (by omega) hnext
    refine ⟨t,?_,hfinal,hf.trans hframe⟩
    rw [ControlView.algorithm_nat_step "start_n" (cursor lower BN i) upper BN _ s (by omega) ((RealLoop.cursor_lt lower upper BN i hN).mpr h),hs,Option.bind_some]
    simpa only [RealLoop.cursor_succ] using ht
  · have he : i = count lower upper BN := by omega
    subst i
    exact ⟨s,ControlView.algorithm_nat_done "start_n" (cursor lower BN (count lower upper BN)) upper BN _ s (by
      have ht : cursor lower BN (count lower upper BN) < upper ↔ count lower upper BN < count lower upper BN :=
        RealLoop.cursor_lt lower upper BN (count lower upper BN) hN
      omega),hr,Frame.refl _ _⟩
termination_by count lower upper BN-i

theorem executes (mem s : BlockState) (c : Config) (BM BN BD lower upper : Nat)
    (second firstDiagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (hr : Registers mem s c BM BN BD lower second seed) :
    ∃ t, stepForRangeIntAux "start_n" (lower : Int) (upper : Int) BN
        (sourceIteration BM BN BD c.p.headDim second firstDiagonal) s = some t ∧
      Registers mem t c BM BN BD (cursor lower BN (count lower upper BN)) second
        (after mem c BM BN BD lower second firstDiagonal seed (count lower upper BN)) ∧ Frame (writes second) s t := by
  simpa only [cursor,RealLoop.cursor,Nat.zero_mul,Nat.add_zero] using
    loop_from mem s c BM BN BD lower upper second firstDiagonal seed hN 0 (Nat.zero_le _)
      (by simpa only [after,closed_zero,cursor,RealLoop.cursor,Nat.zero_mul,Nat.add_zero] using hr)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Loop
