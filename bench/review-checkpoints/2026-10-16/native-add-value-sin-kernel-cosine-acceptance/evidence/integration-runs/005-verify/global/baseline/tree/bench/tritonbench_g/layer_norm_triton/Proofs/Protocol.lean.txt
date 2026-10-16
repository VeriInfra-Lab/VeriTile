import bench.tritonbench_g.layer_norm_triton.Proofs.NativeBoundary

namespace VeriTile.Bench.TritonBenchG.LayerNormLock

def Critical : Phase → Prop
  | .count | .setCount | .loadDW _ | .loadDB _ | .storeDW _ | .storeDB _ | .release => True
  | _ => False

/-- Exact SC mutual exclusion property, including the whole Count/partial
read/write interval. It is not a native memory-model or progress theorem. -/
structure ProtocolInvariant (c : Config) (s : Control) : Prop where
  binary : BinaryWords s.words
  held : ∀ p, Critical (s.pc p) → s.words (lockAddr c p) = 1
  unique : ∀ p q, Critical (s.pc p) → Critical (s.pc q) → lockAddr c p = lockAddr c q → p = q

private theorem enter_critical (c : Config) (p : Nat) (s : Control)
    (hb : BinaryWords s.words) (hn : Critical (decode c p s).2) (ho : ¬Critical (s.pc p)) :
    (decode c p s).1 = .poll (lockAddr c p) ∧ s.words (lockAddr c p) = 0 := by
  have hw := hb (lockAddr c p)
  rcases hw with hw | hw <;> cases hpc : s.pc p <;>
    simp_all [decode, Critical]
  all_goals split_ifs at * <;> simp_all [Critical]

private theorem release_critical (c : Config) (p : Nat) (s : Control) (a : Int)
    (h : (decode c p s).1 = .release a) :
    Critical (s.pc p) ∧ ¬Critical (decode c p s).2 ∧ a = lockAddr c p := by
  cases hpc : s.pc p <;> simp_all [decode, Critical]
  all_goals split_ifs at * <;> simp_all [Critical]

private theorem integer_preserves_held (e : Event) (words : Int → I32) (a : Int)
    (held : words a = 1) (hn : e ≠ .release a) : integerEffect e words a = 1 := by
  cases e <;> simp_all [integerEffect, put, cas01]
  all_goals aesop

/-- One exact source control step preserves mutual exclusion. No fairness,
ready-lock, bounded retry or data-value premise appears. -/
theorem protocol_step (c : Config) (p : Nat) (s : Control) (h : ProtocolInvariant c s) :
    ProtocolInvariant c (controlStep c p s).2 := by
  have selected (hn : Critical (decode c p s).2) (q : Nat) (hq : q ≠ p)
      (hc : Critical (s.pc q)) (ha : lockAddr c p = lockAddr c q) : False := by
    by_cases hp : Critical (s.pc p)
    · exact hq (h.unique p q hp hc ha).symm
    · have hz := (enter_critical c p s h.binary hn hp).2
      have ho := h.held q hc
      rw [ha, ho] at hz
      contradiction
  refine ⟨control_step_binary c p s h.binary, ?_, ?_⟩
  · intro q hq
    by_cases eqp : q = p
    · subst q
      have hn : Critical (decode c p s).2 := by simpa only [controlStep, put_same] using hq
      by_cases old : Critical (s.pc p)
      · have nr : (decode c p s).1 ≠ .release (lockAddr c p) := by
          intro hr; exact (release_critical c p s _ hr).2.1 hn
        exact integer_preserves_held (decode c p s).1 s.words _ (h.held p old) nr
      · have ent := enter_critical c p s h.binary hn old
        change integerEffect (decode c p s).1 s.words (lockAddr c p) = 1
        rw [ent.1]
        simp [integerEffect, put, cas01, ent.2]
    · have old : Critical (s.pc q) := by simpa only [controlStep, put_other _ p q _ eqp] using hq
      have nr : (decode c p s).1 ≠ .release (lockAddr c q) := by
        intro hr
        have rel := release_critical c p s _ hr
        have pq := h.unique p q rel.1 old rel.2.2.symm
        exact eqp pq.symm
      exact integer_preserves_held (decode c p s).1 s.words _ (h.held q old) nr
  · intro q r hq hr same
    by_cases qp : q = p <;> by_cases rp : r = p
    · exact qp.trans rp.symm
    · subst q
      have hn : Critical (decode c p s).2 := by simpa only [controlStep, put_same] using hq
      have oldr : Critical (s.pc r) := by simpa only [controlStep, put_other _ p r _ rp] using hr
      exact False.elim (selected hn r rp oldr same)
    · subst r
      have hn : Critical (decode c p s).2 := by simpa only [controlStep, put_same] using hr
      have oldq : Critical (s.pc q) := by simpa only [controlStep, put_other _ p q _ qp] using hq
      exact False.elim (selected hn q qp oldq same.symm)
    · have oldq : Critical (s.pc q) := by simpa only [controlStep, put_other _ p q _ qp] using hq
      have oldr : Critical (s.pc r) := by simpa only [controlStep, put_other _ p r _ rp] using hr
      exact h.unique q r oldq oldr same

theorem protocol_run (c : Config) (schedule : List Nat) (s : Control)
    (h : ProtocolInvariant c s) : ProtocolInvariant c (runControl c schedule s).2 := by
  induction schedule generalizing s with
  | nil => exact h
  | cons p ps ih => exact ih _ (protocol_step c p s h)

theorem zero_initialization_mutual_exclusion (c : Config) (schedule : List Nat) :
    ProtocolInvariant c (runControl c schedule (initialControl (fun _ => 0))).2 := by
  apply protocol_run
  exact ⟨fun _ => Or.inl rfl, fun _ h => False.elim h, fun _ _ h => False.elim h⟩

#axiomsClean protocol_step
#axiomsClean zero_initialization_mutual_exclusion
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock
