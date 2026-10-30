import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Access

namespace VeriTile.Bench.TritonBenchG.SpinningLock

/-- Before the first store to a particular active lane. This PC invariant is
used contrapositively; completion is never postulated as an output equation. -/
def BeforeOutput (ctl : Control) (lane : Nat) : Prop :=
  ctl.round = 1 ∧ match ctl.phase with
    | .output n => n ≤ lane
    | .done => False
    | _ => True

theorem step_beforeOutput (b : Backend V) (c : Config) (s : State V)
    (p l q : Nat) (hp : p < c.grid) (hl : l < c.lanes)
    (hm : c.outputMask p l = true) (hs : BeforeOutput (s.control p) l)
    (he : (step b c q s).1 ≠ Event.output p l (c.cAddr p l)) :
    BeforeOutput ((step b c q s).2.control p) l := by
  by_cases hpq : p = q
  · subst q
    have hg : ¬p ≥ c.grid := by omega
    rcases hs with ⟨hr, hs⟩
    cases hphase : (s.control p).phase with
    | branch =>
      simp only [step, decode, if_neg hg, hphase, hr, show ¬1 ≥ 10 by decide, if_false, put_same]
      split_ifs <;> simp [BeforeOutput]
    | scan next =>
      simp only [step, decode, if_neg hg, hphase, hr, put_same]
      split_ifs <;> simp [BeforeOutput]
    | poll next =>
      simp only [step, decode, if_neg hg, hphase, hr, put_same]
      split_ifs <;> simp [BeforeOutput]
    | gather next n =>
      simp only [step, decode, if_neg hg, hphase, hr, put_same]
      split_ifs <;> simp [BeforeOutput]
    | publish n =>
      simp only [step, decode, if_neg hg, hphase, hr, put_same]
      split_ifs <;> simp [BeforeOutput]
    | release => simp [step, decode, hg, hphase, hr, BeforeOutput]
    | output n =>
      simp only [hphase] at hs
      have hn : n < c.lanes := by omega
      have hnl : n ≠ l := by
        intro hh
        subst n
        exact he (by simp [step, decode, hg, hphase, hl, hm])
      simp only [step, decode, if_neg hg, hphase, if_pos hn, put_same, BeforeOutput, hr]
      exact ⟨True.intro, by omega⟩
    | done => simp [hphase] at hs
    | fault => simp [step, decode, hg, hphase, hr, BeforeOutput]
  · simpa [step, put, hpq] using hs

theorem run_beforeOutput (b : Backend V) (c : Config) (schedule : List Nat)
    (s : State V) (p l : Nat) (hp : p < c.grid) (hl : l < c.lanes)
    (hm : c.outputMask p l = true) (hs : BeforeOutput (s.control p) l)
    (he : Event.output p l (c.cAddr p l) ∉ (run b c schedule s).1) :
    BeforeOutput ((run b c schedule s).2.control p) l := by
  induction schedule generalizing s with
  | nil => exact hs
  | cons q qs ih =>
    have hh : (step b c q s).1 ≠ Event.output p l (c.cAddr p l) := by
      intro hh; exact he (List.mem_cons.mpr (Or.inl hh.symm))
    have ht : Event.output p l (c.cAddr p l) ∉ (run b c qs (step b c q s).2).1 := by
      intro ht; exact he (List.mem_cons_of_mem _ ht)
    exact ih _ (step_beforeOutput b c s p l q hp hl hm hs hh) ht

theorem completed_lane_visited (b : Backend V) (c : Config) (schedule : List Nat)
    (mem : Memory V) (p l : Nat) (hp : p < c.grid) (hl : l < c.lanes)
    (hm : c.outputMask p l = true)
    (hdone : ((run b c schedule (initial b mem)).2.control p).phase = .done) :
    Event.output p l (c.cAddr p l) ∈ (run b c schedule (initial b mem)).1 := by
  by_contra hn
  have hb := run_beforeOutput b c schedule (initial b mem) p l hp hl hm
    (by simp [BeforeOutput, initial]) hn
  simp [BeforeOutput, hdone] at hb

theorem completed_footprint (h : HostArgs) (b : Backend V) (schedule : List Nat)
    (mem : Memory V) (hdone : allDone h.config (run b h.config schedule (initial b mem)).2 = true)
    (a : Int) :
    Visited (run b h.config schedule (initial b mem)).1 a ↔ h.outputFootprint a := by
  constructor
  · rintro ⟨p, l, he⟩
    obtain ⟨hp, hl, hm, ha⟩ := run_output_origin h.config b schedule (initial b mem) p l a he
    exact ⟨p, hp, l, hl, hm, ha⟩
  · rintro ⟨p, hp, l, hl, hm, rfl⟩
    refine ⟨p, l, completed_lane_visited b h.config schedule mem p l hp hl hm ?_⟩
    have hh := (List.all_eq_true.mp hdone) p (List.mem_range.mpr hp)
    simpa using hh

#axiomsClean completed_lane_visited
#axiomsClean completed_footprint
end VeriTile.Bench.TritonBenchG.SpinningLock
