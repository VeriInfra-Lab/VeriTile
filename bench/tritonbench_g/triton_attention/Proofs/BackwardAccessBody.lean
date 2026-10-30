import bench.tritonbench_g.triton_attention.Proofs.BackwardAccessFragments

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000

theorem backward_body_safe (rest : Algebra α) (P : Primitives α)
    (p : BackwardParameters) (initial current : State α) (block query : Nat)
    (mem : Memory α) (acc : BackwardInner α p)
    (inv : BackwardInnerInvariant p initial block query mem acc current) (bounds : RegionName→Nat)
    (queries : ∀ i : BLane p,backwardOffset p initial query i<bounds p.q)
    (maximum : ∀ i : BRow p,backwardRowOffset p initial query i<bounds p.maximum)
    (gradient : ∀ i : BLane p,backwardOffset p initial query i<bounds p.gradient)
    (delta : ∀ i : BRow p,backwardRowOffset p initial query i<bounds p.delta)
    (dq : ∀ i : BLane p,backwardOffset p initial query i<bounds p.dq) :
    Safety.traceSafeList (backend rest P) bounds (backwardInnerBody p)
      (current.setReg "start_m" .nat [] (fun _=>query*p.blockM)) := by
  let a := backwardScoreResult P p initial current block query mem acc
  let b := backwardGradientResult P p initial a block query mem acc
  have hs := backwardScoreExecution rest P p initial current block query mem acc inv
  have hg := backwardGradientExecution rest P p initial a block query mem acc
    (backwardScoreInvariant P p initial current block query mem acc inv)
    (by simp [a,backwardScoreResult]) (by simp [a,backwardScoreResult])
  have safeGradient := backward_gradient_safe rest P p initial a block query mem acc
    (backwardScoreInvariant P p initial current block query mem acc inv)
    (by simp [a,backwardScoreResult]) (by simp [a,backwardScoreResult]) bounds gradient delta
  have safeStore := backward_store_safe rest P p initial b block query mem acc
    (by simp [b,a,backwardGradientResult,backwardScoreResult,inv.dqPtr])
    (by simp [b,a,backwardGradientResult,backwardScoreResult,inv.key])
    (by simp [b,a,backwardGradientResult,backwardScoreResult])
    (by simp [b,a,backwardGradientResult,backwardScoreResult,inv.dk])
    (by simp [b,backwardGradientResult]) bounds dq
  have split : backwardInnerBody p=(backwardInnerBody p).take 6 ++
      ((backwardInnerBody p).drop 6).take 6 ++ ((backwardInnerBody p).drop 12).take 4 ++
      (backwardInnerBody p).drop 16 := rfl
  rw [split,Safety.traceSafeList_append,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨⟨backward_score_safe rest P p initial current block query mem acc inv bounds queries maximum,?_⟩,?_⟩,?_⟩
  · intro next ran
    rw [hs] at ran
    cases ran
    exact safeGradient
  · intro next ran
    rw [run_append,hs] at ran
    simp only [Option.bind_some] at ran
    have equal : b=next := Option.some.inj (hg.symm.trans ran)
    subst next
    exact safeStore
  · intro next _
    exact backward_advance_safe (backend rest P) p next bounds

#axiomsClean backward_body_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
