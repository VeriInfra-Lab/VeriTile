import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Source

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section

theorem Allocation.window_inside (a : Allocation) (offset count point : Nat)
    (bound : offset+count≤a.length)
    (hit : a.base+offset≤point ∧ point<a.base+offset+count) : a.Contains point := by
  dsimp [Allocation.Contains]
  omega

theorem Shape.main_injective (c : Shape) (head row col head' row' col' : Nat)
    (hr : row<c.T) (hc : col<c.D) (hr' : row'<c.T) (hc' : col'<c.D)
    (he : c.mainOffset head row col=c.mainOffset head' row' col') :
    head=head' ∧ row=row' ∧ col=col' := by
  have he' : (head*c.T+row)*c.D+col=(head'*c.T+row')*c.D+col' := by
    simpa [Shape.mainOffset,Shape.headSize,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using he
  obtain ⟨hpair,hcol⟩ := digit_injective c.D _ _ _ _ hc hc' he'
  obtain ⟨hhead,hrow⟩ := digit_injective c.T _ _ _ _ hr hr' hpair
  exact ⟨hhead,hrow,hcol⟩

namespace Source
theorem active_le (c : Shape) (chunk : Nat) : activeRows c chunk≤32 := Nat.min_le_left _ _

theorem output_end (c : Shape) (head chunk : Nat) (hh : head<c.heads) (hc : chunk<c.chunks) :
    head*c.headSize+chunk*32*c.D+activeRows c chunk*c.D≤c.mainSize := by
  have hs := c.chunk_start chunk hc
  have ha := Nat.min_le_right 32 (c.T-chunk*32)
  have hr : chunk*32+activeRows c chunk≤c.T := by dsimp [activeRows]; omega
  have hm := Nat.mul_le_mul_right c.D hr
  have hh' := Nat.mul_le_mul_right c.headSize (Nat.succ_le_iff.mpr hh)
  simp only [Nat.add_mul,Nat.succ_mul] at hm hh'
  dsimp [Shape.mainSize,Shape.headSize] at *
  omega

theorem history_end (c : Shape) (head chunk : Nat) (hh : head<c.heads) (hc : chunk<c.chunks) :
    head*c.historyHead+chunk*c.D*c.D+c.D*c.D≤c.historySize := by
  have hm := Nat.mul_le_mul_right c.D (Nat.mul_le_mul_right c.D (Nat.succ_le_iff.mpr hc))
  have hh' := Nat.mul_le_mul_right c.historyHead (Nat.succ_le_iff.mpr hh)
  simp only [Nat.succ_mul,Nat.add_mul] at hm hh'
  dsimp [Shape.historySize,Shape.historyHead] at *
  omega

theorem output_window (c : Shape) (abi : ABI c) (head chunk point : Nat)
    (hh : head<c.heads) (hc : chunk<c.chunks)
    (hit : outputBase c abi head chunk≤point ∧
      point<outputBase c abi head chunk+activeRows c chunk*c.D) : abi.o.Contains point := by
  apply Allocation.window_inside abi.o (head*c.headSize+chunk*32*c.D) (activeRows c chunk*c.D) point
  · rw [abi.oLength]
    exact output_end c head chunk hh hc
  · simpa [outputBase,Nat.add_assoc] using hit

theorem history_window (c : Shape) (abi : ABI c) (head chunk point : Nat)
    (hh : head<c.heads) (hc : chunk<c.chunks)
    (hit : historyBase c abi head chunk≤point ∧ point<historyBase c abi head chunk+c.D*c.D) :
    abi.h.Contains point := by
  apply Allocation.window_inside abi.h (head*c.historyHead+chunk*c.D*c.D) (c.D*c.D) point
  · rw [abi.hLength]
    exact history_end c head chunk hh hc
  · simpa [historyBase,Nat.add_assoc] using hit

theorem save_frame (ops : Ops α) (c : Shape) (abi : ABI c) (store : Bool)
    (head chunk : Nat) (s : State α) (hh : head<c.heads) (hc : chunk<c.chunks)
    (point : Nat) (outside : store=Bool.false ∨ ¬abi.h.Contains point) :
    save ops c abi store head chunk s point=s.memory point := by
  cases store with
  | «false» => rfl
  | «true» =>
    simp only [save, Bool.true_eq, ↓reduceIte]
    apply Memory.store_frame
    intro hit
    exact (outside.resolve_left (by decide)) (history_window c abi head chunk point hh hc hit)

theorem step_frame (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head chunk : Nat) (s : State α)
    (hh : head<c.heads) (hc : chunk<c.chunks) (point : Nat)
    (hout : ¬abi.o.Contains point) (hhist : store=Bool.false ∨ ¬abi.h.Contains point) :
    (step ops c abi scale store ifcond head chunk s).memory point=s.memory point := by
  dsimp only [step]
  rw [Memory.store_frame]
  · exact save_frame ops c abi store head chunk s hh hc point hhist
  · intro hit
    exact hout (output_window c abi head chunk point hh hc hit)
end Source

def ReadOnly (c : Shape) (abi : ABI c) (current initial : Memory α) : Prop :=
  ∀ a∈[abi.q,abi.k,abi.v], ∀ point, a.Contains point → current point=initial point

theorem ReadOnly.refl (c : Shape) (abi : ABI c) (m : Memory α) : ReadOnly c abi m m := by
  intro _ _ _ _
  rfl

theorem ReadOnly.trans {c : Shape} {abi : ABI c} {a b d : Memory α}
    (hab : ReadOnly c abi a b) (hbd : ReadOnly c abi b d) : ReadOnly c abi a d := by
  intro r hr p hp
  exact (hab r hr p hp).trans (hbd r hr p hp)

theorem Source.save_readOnly (ops : Ops α) (c : Shape) (abi : ABI c) (store : Bool)
    (head chunk : Nat) (s : Source.State α) (hh : head<c.heads) (hc : chunk<c.chunks) :
    ReadOnly c abi (Source.save ops c abi store head chunk s) s.memory := by
  intro a ha point hp
  apply Source.save_frame ops c abi store head chunk s hh hc point
  right
  intro hhit
  exact abi.historySeparate a (by simpa using List.mem_append_left [abi.o] ha) point hhit hp

theorem Source.step_readOnly (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head chunk : Nat) (s : Source.State α)
    (hh : head<c.heads) (hc : chunk<c.chunks) :
    ReadOnly c abi (Source.step ops c abi scale store ifcond head chunk s).memory s.memory := by
  intro a ha point hp
  apply Source.step_frame ops c abi scale store ifcond head chunk s hh hc point
  · intro hhit
    exact abi.outputSeparate a (by simpa using List.mem_append_left [abi.h] ha) point hhit hp
  · right
    intro hhit
    exact abi.historySeparate a (by simpa using List.mem_append_left [abi.o] ha) point hhit hp

#axiomsClean Source.output_window
#axiomsClean Source.history_window
#axiomsClean Source.step_readOnly
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
