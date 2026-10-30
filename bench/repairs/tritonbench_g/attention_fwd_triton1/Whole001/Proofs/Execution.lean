import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Memory

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section

theorem Allocation.contains_offset (a : Allocation) (offset : Nat) (h : offset<a.length) :
    a.Contains (a.base+offset) := by
  dsimp [Allocation.Contains]
  omega

theorem ReadOnly.masked_equal (ops : Ops α) (c : Shape) (abi : ABI c)
    (current initial : Memory α) (ro : ReadOnly c abi current initial)
    (a : Allocation) (ha : a∈[abi.q,abi.k,abi.v]) (hlen : a.length=c.mainSize)
    (head : Nat) (hh : head<c.heads) (row col : Nat) :
    masked ops c (fun h r d => current (a.base+c.mainOffset h r d)) head row col =
      masked ops c (fun h r d => initial (a.base+c.mainOffset h r d)) head row col := by
  unfold masked
  split_ifs with h
  · apply ro a ha
    apply Allocation.contains_offset
    rw [hlen]
    exact c.main_bound head row col hh h.1 h.2
  · rfl

theorem Body.evaluate_readOnly (ops : Ops α) (c : Shape) (abi : ABI c)
    (current initial : Memory α) (ro : ReadOnly c abi current initial)
    (scale : α) (head chunk : Nat) (hh : head<c.heads) (carry : Tile α) (ifcond : Bool) :
    Body.evaluate ops c (inputsOf c abi current) scale head chunk carry ifcond =
      Body.evaluate ops c (inputsOf c abi initial) scale head chunk carry ifcond := by
  have hq : Body.query ops c (inputsOf c abi current) scale head chunk =
      Body.query ops c (inputsOf c abi initial) scale head chunk := by
    funext i d
    dsimp [Body.query,inputsOf]
    rw [ro.masked_equal ops c abi current initial abi.q (by simp) abi.qLength head hh]
  have hk : Body.key ops c (inputsOf c abi current) head chunk =
      Body.key ops c (inputsOf c abi initial) head chunk := by
    funext d i
    exact ro.masked_equal ops c abi current initial abi.k (by simp) abi.kLength head hh _ _
  have hv : Body.value ops c (inputsOf c abi current) head chunk =
      Body.value ops c (inputsOf c abi initial) head chunk := by
    funext i d
    exact ro.masked_equal ops c abi current initial abi.v (by simp) abi.vLength head hh _ _
  simp only [Body.evaluate,hq,hk,hv]

namespace Source
theorem step_values (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head chunk : Nat) (s : State α) (initial : Memory α)
    (hh : head<c.heads) (hc : chunk<c.chunks) (ro : ReadOnly c abi s.memory initial) :
    Body.evaluate ops c (inputsOf c abi (save ops c abi store head chunk s))
        scale head chunk s.carry ifcond =
      Body.evaluate ops c (inputsOf c abi initial) scale head chunk s.carry ifcond :=
  Body.evaluate_readOnly ops c abi _ initial
    ((save_readOnly ops c abi store head chunk s hh hc).trans ro) scale head chunk hh _ ifcond

theorem step_carry (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head chunk : Nat) (s : State α) (initial : Memory α)
    (hh : head<c.heads) (hc : chunk<c.chunks) (ro : ReadOnly c abi s.memory initial)
    (carry : s.carry=Reference.carry ops c (inputsOf c abi initial) scale head ifcond chunk) :
    (step ops c abi scale store ifcond head chunk s).carry =
      Reference.carry ops c (inputsOf c abi initial) scale head ifcond (chunk+1) := by
  dsimp only [step,Reference.carry]
  rw [step_values ops c abi scale store ifcond head chunk s initial hh hc ro,carry]

theorem lane_div (c : Shape) (i d : Nat) (hd : d<c.D) : (i*c.D+d)/c.D=i := by
  rw [Nat.add_comm (i*c.D) d,Nat.add_mul_div_right d i c.dim_pos,Nat.div_eq_of_lt hd,Nat.zero_add]
theorem lane_mod (c : Shape) (i d : Nat) (hd : d<c.D) : (i*c.D+d)%c.D=d := by
  simp [Nat.add_mod,Nat.mod_eq_of_lt hd]

theorem step_output (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head chunk : Nat) (s : State α) (initial : Memory α)
    (hh : head<c.heads) (hc : chunk<c.chunks) (ro : ReadOnly c abi s.memory initial)
    (carry : s.carry=Reference.carry ops c (inputsOf c abi initial) scale head ifcond chunk)
    (i d : Nat) (hi : i<activeRows c chunk) (hd : d<c.D) :
    (step ops c abi scale store ifcond head chunk s).memory
      (abi.o.base+c.mainOffset head (chunk*32+i) d) =
      ops.cast (Reference.output ops c (inputsOf c abi initial) scale head chunk ifcond i d) := by
  have addr : abi.o.base+c.mainOffset head (chunk*32+i) d=
      outputBase c abi head chunk+(i*c.D+d) := by
    simp [Shape.mainOffset,outputBase,Nat.add_mul,Nat.add_assoc]
  dsimp only [step]
  rw [addr,Memory.store_read _ _ _ _ (digit_bound i _ d _ hi hd)]
  rw [lane_div c i d hd,lane_mod c i d hd,
    step_values ops c abi scale store ifcond head chunk s initial hh hc ro,carry]
  rfl

theorem step_history (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head chunk : Nat) (s : State α)
    (hh : head<c.heads) (hc : chunk<c.chunks) (hs : store=Bool.true)
    (i d : Nat) (hi : i<c.D) (hd : d<c.D) :
    (step ops c abi scale store ifcond head chunk s).memory
      (abi.h.base+c.historyOffset head chunk i d) = ops.cast (s.carry i d) := by
  have hp : abi.h.Contains (abi.h.base+c.historyOffset head chunk i d) := by
    apply Allocation.contains_offset
    rw [abi.hLength]
    exact c.history_bound head chunk i d hh hc hi hd
  have addr : abi.h.base+c.historyOffset head chunk i d=
      historyBase c abi head chunk+(i*c.D+d) := by
    simp [Shape.historyOffset,historyBase,Nat.add_mul,Nat.add_assoc]
  dsimp only [step]
  rw [Memory.store_frame]
  · simp only [save,hs,Bool.true_eq,↓reduceIte]
    rw [addr,Memory.store_read _ _ _ _ (digit_bound i _ d _ hi hd),
      lane_div c i d hd,lane_mod c i d hd]
  · intro hit
    exact abi.outputSeparate abi.h (by simp) _ (output_window c abi head chunk _ hh hc hit) hp

/-- A point in O before this tile survives the H snapshot and the O store. -/
theorem step_output_before (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head chunk : Nat) (s : State α)
    (hh : head<c.heads) (hc : chunk<c.chunks) (point : Nat)
    (hp : abi.o.Contains point) (before : point<outputBase c abi head chunk) :
    (step ops c abi scale store ifcond head chunk s).memory point=s.memory point := by
  dsimp only [step]
  rw [Memory.store_frame _ _ _ _ _ (by omega)]
  apply save_frame ops c abi store head chunk s hh hc point
  right
  intro hit
  exact abi.historySeparate abi.o (by simp) point hit hp

theorem step_history_before (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head chunk : Nat) (s : State α)
    (hh : head<c.heads) (hc : chunk<c.chunks) (point : Nat)
    (hp : abi.h.Contains point) (before : point<historyBase c abi head chunk) :
    (step ops c abi scale store ifcond head chunk s).memory point=s.memory point := by
  dsimp only [step]
  rw [Memory.store_frame]
  · cases store with
    | «false» => rfl
    | «true» =>
      simp only [save,Bool.true_eq,↓reduceIte]
      exact Memory.store_frame _ _ _ _ _ (by omega)
  · intro hit
    exact abi.outputSeparate abi.h (by simp) point (output_window c abi head chunk _ hh hc hit) hp

end Source

#axiomsClean Source.step_output
#axiomsClean Source.step_history
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
