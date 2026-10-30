import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Loop

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section
namespace Source

theorem active_row (c : Shape) (row : Nat) (hr : row<c.T) : row%32<activeRows c (row/32) := by
  have hmod := Nat.mod_lt row (by decide : 0<32)
  have he := Nat.div_add_mod row 32
  dsimp [activeRows]
  omega

theorem output_head_earlier (c : Shape) (abi : ABI c) (head later row d : Nat)
    (hh : head<later) (hr : row<c.T) (hd : d<c.D) :
    abi.o.base+c.mainOffset head row d<outputBase c abi later 0 := by
  have hrow := digit_bound row c.T d c.D hr hd
  have he := digit_bound head later (row*c.D+d) c.headSize hh hrow
  dsimp [Shape.mainOffset,outputBase]
  omega

theorem history_head_earlier (c : Shape) (abi : ABI c) (head later chunk i d : Nat)
    (hh : head<later) (hc : chunk<c.chunks) (hi : i<c.D) (hd : d<c.D) :
    abi.h.base+c.historyOffset head chunk i d<historyBase c abi later 0 := by
  have hrow := digit_bound chunk c.chunks i c.D hc hi
  have hcell := digit_bound (chunk*c.D+i) (c.chunks*c.D) d c.D hrow hd
  have he := digit_bound head later ((chunk*c.D+i)*c.D+d) c.historyHead hh hcell
  dsimp [Shape.historyOffset,historyBase]
  omega

variable (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (initial : Memory α)

theorem grid_readOnly (n : Nat) : n≤c.heads →
    ReadOnly c abi (grid ops c abi scale store ifcond initial n) initial := by
  induction n with
  | zero => intro _; exact ReadOnly.refl c abi initial
  | succ n ih =>
    intro hn
    exact loop_readOnly ops c abi scale store ifcond n _ initial (by omega)
      (ih (by omega)) c.chunks (by rfl)

theorem grid_output (n : Nat) : n≤c.heads → ∀ head, head<n → ∀ row d,
    row<c.T → d<c.D →
    grid ops c abi scale store ifcond initial n (abi.o.base+c.mainOffset head row d)=
      ops.cast (Reference.output ops c (inputsOf c abi initial) scale head (row/32) ifcond (row%32) d) := by
  induction n with
  | zero => intro _ head hh; omega
  | succ n ih =>
    intro hn head hh row d hr hd
    by_cases hhn : head<n
    · rw [grid,kernel,loop_output_before]
      · exact ih (by omega) head hhn row d hr hd
      · omega
      · apply Allocation.contains_offset
        rw [abi.oLength]
        exact c.main_bound head row d (by omega) hr hd
      · exact output_head_earlier c abi head n row d hhn hr hd
      · rfl
    · have he : head=n := by omega
      subst head
      have hm : row/32*32+row%32=row := by omega
      have hout := loop_output ops c abi scale store ifcond n
        (grid ops c abi scale store ifcond initial n) initial (by omega)
        (grid_readOnly ops c abi scale store ifcond initial n (by omega))
        c.chunks (by rfl) (row/32) (c.chunk_of_row row hr) (row%32) d (active_row c row hr) hd
      simpa only [grid,kernel,hm] using hout

theorem grid_history (hs : store=Bool.true) (n : Nat) : n≤c.heads →
    ∀ head, head<n → ∀ chunk i d, chunk<c.chunks → i<c.D → d<c.D →
    grid ops c abi scale store ifcond initial n (abi.h.base+c.historyOffset head chunk i d)=
      ops.cast (Reference.carry ops c (inputsOf c abi initial) scale head ifcond chunk i d) := by
  induction n with
  | zero => intro _ head hh; omega
  | succ n ih =>
    intro hn head hh chunk i d hc hi hd
    by_cases hhn : head<n
    · rw [grid,kernel,loop_history_before]
      · exact ih (by omega) head hhn chunk i d hc hi hd
      · omega
      · apply Allocation.contains_offset
        rw [abi.hLength]
        exact c.history_bound head chunk i d (by omega) hc hi hd
      · exact history_head_earlier c abi head n chunk i d hhn hc hi hd
      · rfl
    · have he : head=n := by omega
      subst head
      exact loop_history ops c abi scale store ifcond n _ initial (by omega)
        (grid_readOnly ops c abi scale store ifcond initial n (by omega))
        hs c.chunks (by rfl) chunk hc i d hi hd

theorem grid_frame (point : Nat) (ho : ¬abi.o.Contains point)
    (hst : store=Bool.false ∨ ¬abi.h.Contains point) (n : Nat) : n≤c.heads →
    grid ops c abi scale store ifcond initial n point=initial point := by
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
    intro hn
    rw [grid,kernel,loop_frame ops c abi scale store ifcond n _ (by omega) point ho hst c.chunks (by rfl)]
    exact ih (by omega)

theorem run_output (head row d : Nat) (hh : head<c.heads) (hr : row<c.T) (hd : d<c.D) :
    run ops c abi scale store ifcond initial (abi.o.base+c.mainOffset head row d)=
      ops.cast (Reference.output ops c (inputsOf c abi initial) scale head (row/32) ifcond (row%32) d) := by
  rw [run,if_neg (by omega)]
  exact grid_output ops c abi scale store ifcond initial c.heads (by rfl) head hh row d hr hd

theorem run_history (hs : store=Bool.true) (head chunk i d : Nat)
    (hh : head<c.heads) (hc : chunk<c.chunks) (hi : i<c.D) (hd : d<c.D) :
    run ops c abi scale store ifcond initial (abi.h.base+c.historyOffset head chunk i d)=
      ops.cast (Reference.carry ops c (inputsOf c abi initial) scale head ifcond chunk i d) := by
  have := c.chunk_start chunk hc
  rw [run,if_neg (by omega)]
  exact grid_history ops c abi scale store ifcond initial hs c.heads (by rfl) head hh chunk i d hc hi hd

theorem run_frame (point : Nat) (ho : ¬abi.o.Contains point)
    (hst : store=Bool.false ∨ ¬abi.h.Contains point) :
    run ops c abi scale store ifcond initial point=initial point := by
  unfold run
  split_ifs
  · rfl
  · exact grid_frame ops c abi scale store ifcond initial point ho hst c.heads (by rfl)

theorem run_readOnly : ReadOnly c abi (run ops c abi scale store ifcond initial) initial := by
  unfold run
  split_ifs
  · exact ReadOnly.refl c abi initial
  · exact grid_readOnly ops c abi scale store ifcond initial c.heads (by rfl)

theorem run_history_disabled (hs : store=Bool.false) (point : Nat) (hp : abi.h.Contains point) :
    run ops c abi scale store ifcond initial point=initial point := by
  apply run_frame ops c abi scale store ifcond initial point
  · intro hit
    exact abi.outputSeparate abi.h (by simp) point hit hp
  · exact Or.inl hs

end Source
#axiomsClean Source.run_output
#axiomsClean Source.run_history
#axiomsClean Source.run_history_disabled
#axiomsClean Source.run_readOnly
#axiomsClean Source.run_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
