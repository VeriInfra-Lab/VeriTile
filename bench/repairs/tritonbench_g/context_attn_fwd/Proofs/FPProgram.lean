import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.FPSuccess

namespace VeriTile.Bench.Repairs.ContextAttnFwd.FPSuccess
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem signed_range_invariant (M : Algebra α) (idx : RegName)
    (body : List ComputeStmt) (P : State α → Prop) (stride : Nat) (hd : 0 < stride)
    (hstep : ∀ i s, P s → ∃ t,
      run M body (s.setReg idx .int [] (fun _ => i)) = some t ∧ P t)
    (cur stop : Int) (s : State α) (hs : P s) :
    ∃ t, rangeInt M idx cur stop stride body s = some t ∧ P t := by
  have key : ∀ k i s, (stop-i).toNat = k → P s →
      ∃ t, rangeInt M idx i stop stride body s = some t ∧ P t := by
    intro k
    induction k using Nat.strong_induction_on with
    | h k ih =>
      intro i s hk hs
      by_cases hi : i < stop
      · obtain ⟨t,ht,hpt⟩ := hstep i s hs
        obtain ⟨u,hu,hpu⟩ := ih (stop-(i+stride)).toNat (by omega)
          (i+stride) t rfl hpt
        refine ⟨u,?_,hpu⟩
        rw [rangeInt]
        simp only [Nat.ne_of_gt hd, ↓reduceIte, hi, ht, Option.bind_some]
        exact hu
      · refine ⟨s,?_,hs⟩
        rw [rangeInt]
        simp [hi]
  exact key (stop-cur).toNat cur s rfl hs

theorem range_success (p : Layout) (r : Regions) (scale : ℝ)
    (M : Algebra α) (hM : DotSupport p M) (hBN : 0 < p.BN)
    (cur stop : Int) (s : State α) (hs : Ready p s) :
    ∃ t, rangeInt (M.withDefaultPrecision .fp32) "start_n" cur stop p.BN
        (parts p r scale).body s = some t ∧ Ready p t ∧
      t.mem = s.mem ∧ Fixed p s t := by
  apply signed_range_invariant (M.withDefaultPrecision .fp32) "start_n"
    (parts p r scale).body (fun t => Ready p t ∧ t.mem = s.mem ∧ Fixed p s t)
    p.BN hBN ?_ cur stop s ⟨hs,rfl,Fixed.refl p s⟩
  intro i t h
  obtain ⟨u,hu,hru,hmu,hfu⟩ := body_success p r scale M hM hBN t h.1 i
  exact ⟨u,hu,hru,hmu.trans h.2.1,h.2.2.trans hfu⟩


end
end VeriTile.Bench.Repairs.ContextAttnFwd.FPSuccess
