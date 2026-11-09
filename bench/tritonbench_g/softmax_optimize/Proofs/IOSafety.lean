import bench.tritonbench_g.softmax_optimize.Proofs.IOSafetyLoops

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.IOSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open IOSafetyLoops IOSafetyBodies
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem after_seed (bounds : RegionBounds) (Y X : RegionName) (N B row : Nat)
    (hN : 0 < N) (hB : 0 < B)
    (hx : ∀ i, i < N → row*N+i < bounds X)
    (hy : ∀ i, i < N → row*N+i < bounds Y)
    (s : BlockState) (hc : Control N B row s) :
    Stmt.TraceSafeList bounds
      ([fullLoop B (Source.inputFull X N B),tailLoop N B (Source.inputTail X N B)] ++
        reductionStmts B ++
        Source.previousCall N B "__veritile_binary_jit_3" "__veritile_binary_jit_4" "__veritile_binary_jit_5" ++ [
          fullLoop B (Source.outputFull Y X N B),tailLoop N B (Source.outputTail Y X N B)]) s := by
  simp only [List.append_assoc,List.cons_append,List.nil_append]
  apply Stmt.TraceSafeList.cons_intro
  · apply full_safe bounds N B row _ s hc
    apply full_range bounds Y X N B row hB _ (by simp [Bodies]) _ 0 s (dvd_zero B) hc
    intro start t inv active
    exact input_full bounds X N B start row t inv.1 (fun i => hx _ (active i))
  · intro t ht
    have hct := full_preserves Y X N B row _ (by simp [Bodies]) s t hc ht
    apply Stmt.TraceSafeList.cons_intro
    · apply tail_safe bounds N B row _ t hct
      apply tail_range bounds Y X N B row hB _ (by simp [Bodies]) _ _ t hct
      intro start u inv
      exact input_tail bounds X N B start row u inv.1 (fun i active => hx _ active)
    · intro u hu
      have hcu := tail_preserves Y X N B row _ (by simp [Bodies]) t u hct hu
      apply TraceProofs.append_safe (reductionStmts B) u (Safety.reduction_safe bounds B u)
      intro v hv
      have hcv := preserves Y X N B row _ (by simp [Bodies]) u v hcu hv
      have previous := Previous.run N B hN hB
        "__veritile_binary_jit_3" "__veritile_binary_jit_4" "__veritile_binary_jit_5" (by decide) v
      apply TraceProofs.append_safe _ v
      · apply TraceProofs.all_safe
        intro st member u
        simp only [Source.previousCall,List.mem_cons,List.not_mem_nil,or_false] at member
        rcases member with rfl | rfl | rfl | rfl <;>
          simp [Stmt.TraceSafe,Source.previousExpression,Op.tritonDiv,Op.truncDiv,Op.SafeAt.eq_def]
      · intro prepared executed
        have same := Option.some.inj (previous.symm.trans executed)
        subst prepared
        have hcp : Control N B row
            (Previous.state v N B "__veritile_binary_jit_3" "__veritile_binary_jit_4" "__veritile_binary_jit_5") := by
          simpa [Control,Previous.state] using hcv.1
        apply Stmt.TraceSafeList.cons_intro
        · apply full_safe bounds N B row _ _ hcp
          apply full_range bounds Y X N B row hB _ (by simp [Bodies]) _ 0 _ (dvd_zero B) hcp
          intro start w inv active
          exact output_full bounds Y X N B start row w inv.1
            (fun i => hx _ (active i)) (fun i => hy _ (active i))
        · intro w hw
          have hcw := full_preserves Y X N B row _ (by simp [Bodies]) _ w hcp hw
          apply Stmt.TraceSafeList.cons_intro
          · apply tail_safe bounds N B row _ w hcw
            apply tail_range bounds Y X N B row hB _ (by simp [Bodies]) _ _ w hcw
            intro start z inv
            exact output_tail bounds Y X N B start row z inv.1
              (fun i active => hx _ active) (fun i active => hy _ active)
          · intro _ _
            exact Stmt.TraceSafeList.nil_intro

/-- The full actual source is safe from exactly its logical input/output row
windows, for any PID. No numerical, success, or output-value premise is used. -/
theorem source_safe (bounds : RegionBounds) (Y X : RegionName) (M N B : Nat)
    (hN : 0 < N) (hB : 0 < B) (s : BlockState)
    (hx : ∀ i, i < N → s.pid*N+i < bounds X)
    (hy : ∀ i, i < N → s.pid*N+i < bounds Y) :
    Kernel.TraceSafe bounds (sourceKernel Y X M N B).eraseDType.toAlgKernel s := by
  unfold Kernel.TraceSafe
  rw [Source.source_body]
  change Stmt.TraceSafeList bounds
    (Reduction.seedStatements N B ++
      ([fullLoop B (Source.inputFull X N B),tailLoop N B (Source.inputTail X N B)] ++
        reductionStmts B ++
        Source.previousCall N B "__veritile_binary_jit_3" "__veritile_binary_jit_4" "__veritile_binary_jit_5" ++ [
          fullLoop B (Source.outputFull Y X N B),tailLoop N B (Source.outputTail Y X N B)])) s
  apply TraceProofs.append_safe (Reduction.seedStatements N B) s
  · apply TraceProofs.all_safe
    intro st member t
    simp only [Reduction.seedStatements,Source.previousCall,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with (rfl | rfl | rfl) | (rfl | rfl | rfl | rfl) <;>
      simp [Stmt.TraceSafe,Source.previousExpression,Op.tritonDiv,Op.truncDiv,Op.SafeAt.eq_def]
  · intro t executed
    have same := Option.some.inj ((Reduction.seed_run s N B hN hB).symm.trans executed)
    subst t
    apply after_seed bounds Y X N B s.pid hN hB hx hy
    simp [Control,Reduction.seed,Previous.state]

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.IOSafety
