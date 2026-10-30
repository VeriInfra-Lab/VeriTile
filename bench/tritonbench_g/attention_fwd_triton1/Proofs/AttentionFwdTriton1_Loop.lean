import bench.tritonbench_g.attention_fwd_triton1.Proofs.Execution

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section
namespace Source

theorem active_valid (c : Shape) (chunk i : Nat) (hi : i<activeRows c chunk) :
    chunk*32+i<c.T := by
  have := Nat.lt_of_lt_of_le hi (Nat.min_le_right 32 (c.T-chunk*32))
  omega

theorem output_earlier (c : Shape) (abi : ABI c) (head chunk later i d : Nat)
    (hc : chunk<later) (hi : i<activeRows c chunk) (hd : d<c.D) :
    abi.o.base+c.mainOffset head (chunk*32+i) d<outputBase c abi head later := by
  have ht := Nat.lt_of_lt_of_le hi (active_le c chunk)
  have hr : chunk*32+i<later*32 := by omega
  have he := digit_bound (chunk*32+i) (later*32) d c.D hr hd
  dsimp [Shape.mainOffset,outputBase]
  omega

theorem history_earlier (c : Shape) (abi : ABI c) (head chunk later i d : Nat)
    (hc : chunk<later) (hi : i<c.D) (hd : d<c.D) :
    abi.h.base+c.historyOffset head chunk i d<historyBase c abi head later := by
  have hrow := digit_bound chunk later i c.D hc hi
  have he := digit_bound (chunk*c.D+i) (later*c.D) d c.D hrow hd
  dsimp [Shape.historyOffset,historyBase]
  omega

variable (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head : Nat) (entry initial : Memory α)

theorem loop_readOnly (hh : head<c.heads) (hentry : ReadOnly c abi entry initial)
    (n : Nat) : n≤c.chunks →
    ReadOnly c abi (loop ops c abi scale store ifcond head entry n).memory initial := by
  induction n with
  | zero => intro _; exact hentry
  | succ n ih =>
    intro hn
    exact (step_readOnly ops c abi scale store ifcond head n _ hh (by omega)).trans (ih (by omega))

theorem loop_carry (hh : head<c.heads) (hentry : ReadOnly c abi entry initial)
    (n : Nat) : n≤c.chunks →
    (loop ops c abi scale store ifcond head entry n).carry=
      Reference.carry ops c (inputsOf c abi initial) scale head ifcond n := by
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
    intro hn
    exact step_carry ops c abi scale store ifcond head n _ initial hh (by omega)
      (loop_readOnly ops c abi scale store ifcond head entry initial hh hentry n (by omega))
      (ih (by omega))

theorem loop_output (hh : head<c.heads) (hentry : ReadOnly c abi entry initial)
    (n : Nat) : n≤c.chunks → ∀ chunk, chunk<n → ∀ i d,
    i<activeRows c chunk → d<c.D →
    (loop ops c abi scale store ifcond head entry n).memory
      (abi.o.base+c.mainOffset head (chunk*32+i) d)=
      ops.cast (Reference.output ops c (inputsOf c abi initial) scale head chunk ifcond i d) := by
  induction n with
  | zero => intro _ chunk hc; omega
  | succ n ih =>
    intro hn chunk hc i d hi hd
    by_cases hcn : chunk<n
    · rw [loop,step_output_before]
      · exact ih (by omega) chunk hcn i d hi hd
      · exact hh
      · omega
      · apply Allocation.contains_offset
        rw [abi.oLength]
        exact c.main_bound head (chunk*32+i) d hh (active_valid c chunk i hi) hd
      · exact output_earlier c abi head chunk n i d hcn hi hd
    · have he : chunk=n := by omega
      subst chunk
      exact step_output ops c abi scale store ifcond head n _ initial hh (by omega)
        (loop_readOnly ops c abi scale store ifcond head entry initial hh hentry n (by omega))
        (loop_carry ops c abi scale store ifcond head entry initial hh hentry n (by omega))
        i d hi hd

theorem loop_history (hh : head<c.heads) (hentry : ReadOnly c abi entry initial)
    (hs : store=Bool.true) (n : Nat) : n≤c.chunks → ∀ chunk, chunk<n → ∀ i d,
    i<c.D → d<c.D →
    (loop ops c abi scale store ifcond head entry n).memory
      (abi.h.base+c.historyOffset head chunk i d)=
      ops.cast (Reference.carry ops c (inputsOf c abi initial) scale head ifcond chunk i d) := by
  induction n with
  | zero => intro _ chunk hc; omega
  | succ n ih =>
    intro hn chunk hc i d hi hd
    by_cases hcn : chunk<n
    · rw [loop,step_history_before]
      · exact ih (by omega) chunk hcn i d hi hd
      · exact hh
      · omega
      · apply Allocation.contains_offset
        rw [abi.hLength]
        exact c.history_bound head chunk i d hh (by omega) hi hd
      · exact history_earlier c abi head chunk n i d hcn hi hd
    · have he : chunk=n := by omega
      subst chunk
      rw [loop,step_history ops c abi scale store ifcond head n _ hh (by omega) hs i d hi hd,
        loop_carry ops c abi scale store ifcond head entry initial hh hentry n (by omega)]

theorem loop_frame (hh : head<c.heads) (point : Nat)
    (ho : ¬abi.o.Contains point) (hst : store=Bool.false ∨ ¬abi.h.Contains point)
    (n : Nat) : n≤c.chunks →
    (loop ops c abi scale store ifcond head entry n).memory point=entry point := by
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
    intro hn
    rw [loop,step_frame ops c abi scale store ifcond head n _ hh (by omega) point ho hst]
    exact ih (by omega)

theorem loop_output_before (hh : head<c.heads) (point : Nat)
    (hp : abi.o.Contains point) (before : point<outputBase c abi head 0)
    (n : Nat) : n≤c.chunks →
    (loop ops c abi scale store ifcond head entry n).memory point=entry point := by
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
    intro hn
    have hb : point<outputBase c abi head n := by dsimp [outputBase] at *; omega
    rw [loop,step_output_before ops c abi scale store ifcond head n _ hh (by omega) point hp hb]
    exact ih (by omega)

theorem loop_history_before (hh : head<c.heads) (point : Nat)
    (hp : abi.h.Contains point) (before : point<historyBase c abi head 0)
    (n : Nat) : n≤c.chunks →
    (loop ops c abi scale store ifcond head entry n).memory point=entry point := by
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
    intro hn
    have hb : point<historyBase c abi head n := by dsimp [historyBase] at *; omega
    rw [loop,step_history_before ops c abi scale store ifcond head n _ hh (by omega) point hp hb]
    exact ih (by omega)

end Source
#axiomsClean Source.loop_output
#axiomsClean Source.loop_history
#axiomsClean Source.loop_readOnly
#axiomsClean Source.loop_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
