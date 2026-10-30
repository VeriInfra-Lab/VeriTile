import bench.tritonbench_g.fused_recurrent_delta.Proofs.RuntimeTransport

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeTransport
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

 theorem range_pair (M : Algebra α) (value : α) (idx : RegName) (hi : idx ≠ "scale")
    (a b : List ComputeStmt) (body : Pair M value a b) (cur stop stride : Nat)
    (s : State α) (hs : Bound value s) :
    FP.Structural.range M idx cur stop stride a s = FP.Structural.range M idx cur stop stride b s ∧
    (∀ bounds, Safety.rangeSafe M bounds idx cur stop stride a s ↔ Safety.rangeSafe M bounds idx cur stop stride b s) ∧
    (∀ t, FP.Structural.range M idx cur stop stride a s=some t → Bound value t) := by
  by_cases hd : stride=0
  · simp [FP.Structural.range,Safety.rangeSafe,hd,hs]
  · by_cases hc : cur<stop
    · have bound : Bound value (s.setReg idx .nat [] (fun _ => cur)) := by
        simpa [Bound,State.setReg,hi,Ne.symm hi] using hs
      have he := body.executes (s.setReg idx .nat [] (fun _ => cur)) bound
      cases hr : run M a (s.setReg idx .nat [] (fun _ => cur)) with
      | none =>
        constructor
        · rw [range_step M idx cur stop stride a s hd hc,range_step M idx cur stop stride b s hd hc,←he,hr]
          rfl
        · constructor
          · intro bounds
            conv_lhs => rw [Safety.rangeSafe_unfold]
            conv_rhs => rw [Safety.rangeSafe_unfold]
            simp only [hd,hc,↓reduceIte,←he,hr,body.safe bounds _ bound]
          · intro t ht
            rw [range_step M idx cur stop stride a s hd hc,hr] at ht
            cases ht
      | some t =>
        have ht := body.keeps _ t bound hr
        have ih := range_pair M value idx hi a b body (cur+stride) stop stride t ht
        constructor
        · rw [range_step M idx cur stop stride a s hd hc,range_step M idx cur stop stride b s hd hc,←he,hr]
          exact ih.1
        · constructor
          · intro bounds
            conv_lhs => rw [Safety.rangeSafe_unfold]
            conv_rhs => rw [Safety.rangeSafe_unfold]
            simp only [hd,hc,↓reduceIte,←he,hr]
            exact and_congr (body.safe bounds _ bound) (ih.2.1 bounds)
          · intro u hu
            rw [range_step M idx cur stop stride a s hd hc,hr] at hu
            exact ih.2.2 u hu
    · simp [FP.Structural.range,Safety.rangeSafe,hd,hc,hs]
termination_by stop-cur

 theorem rangeAtom (M : Algebra α) (value : α) (idx : RegName) (hi : idx ≠ "scale")
    (cur stop stride : Nat) (a b : List ComputeStmt) (body : Pair M value a b) :
    Atom M value (.forRange idx cur stop stride a) (.forRange idx cur stop stride b) := by
  constructor
  · intro s hs; simpa only [step] using (range_pair M value idx hi a b body cur stop stride s hs).1
  · intro bounds s hs; simpa only [Safety.stepSafe] using (range_pair M value idx hi a b body cur stop stride s hs).2.1 bounds
  · intro s t hs ht; exact (range_pair M value idx hi a b body cur stop stride s hs).2.2 t (by simpa only [step] using ht)

 theorem dynamicAtom (M : Algebra α) (value : α) (idx : RegName) (hi : idx ≠ "scale")
    (lo stop stride : Op .nat []) (a b : List ComputeStmt) (body : Pair M value a b) :
    Atom M value (.forRangeDyn idx lo stop stride a) (.forRangeDyn idx lo stop stride b) := by
  constructor
  · intro s hs
    simp only [step]
    cases ha : FP.Structural.evalOp M none lo s <;> cases hb : FP.Structural.evalOp M none stop s <;>
      cases hd : FP.Structural.evalOp M none stride s <;> simp only [bind,pure,Option.bind]
    exact (range_pair M value idx hi a b body _ _ _ s hs).1
  · intro bounds s hs
    simp only [Safety.stepSafe]
    cases ha : FP.Structural.evalOp M none lo s <;> cases hb : FP.Structural.evalOp M none stop s <;>
      cases hd : FP.Structural.evalOp M none stride s <;> simp only [ha,hb,hd]
    all_goals try rfl
    exact and_congr Iff.rfl (and_congr Iff.rfl (and_congr Iff.rfl
      ((range_pair M value idx hi a b body _ _ _ s hs).2.1 bounds)))
  · intro s t hs ht
    cases ha : FP.Structural.evalOp M none lo s <;> cases hb : FP.Structural.evalOp M none stop s <;>
      cases hd : FP.Structural.evalOp M none stride s <;> simp only [step,ha,hb,hd,bind,pure,Option.bind] at ht
    all_goals try contradiction
    exact (range_pair M value idx hi a b body _ _ _ s hs).2.2 t ht

 theorem ifAtom (M : Algebra α) (value : α) (cond : Op .bool [])
    (a b : List ComputeStmt) (body : Pair M value a b) :
    Atom M value (.ifThen cond a) (.ifThen cond b) := by
  constructor
  · intro s hs
    simp only [step]
    cases hc : FP.Structural.evalOp M none cond s with
    | none => rfl
    | some c =>
      simp only [bind,pure,Option.bind]
      by_cases ht : c PUnit.unit=Bool.true
      · simp only [ht,↓reduceIte]; exact body.executes s hs
      · simp only [ht,↓reduceIte]; rfl
  · intro bounds s hs
    simp only [Safety.stepSafe]
    cases hc : FP.Structural.evalOp M none cond s with
    | none => rfl
    | some c =>
      simp only [hc]
      by_cases ht : c PUnit.unit=Bool.true
      · simp only [ht,↓reduceIte]; exact and_congr Iff.rfl (body.safe bounds s hs)
      · simp only [ht,↓reduceIte]; rfl
  · intro s t hs ht
    cases hc : FP.Structural.evalOp M none cond s with
    | none => simp [step,hc] at ht
    | some c =>
      simp only [step,hc,bind,pure,Option.bind] at ht
      split at ht
      · exact body.keeps s t hs ht
      · cases ht; exact hs

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeTransport
