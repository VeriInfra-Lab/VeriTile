import bench.tritonbench_g.rotary_emb.Contract

namespace VeriTile.Bench.Repairs.RotaryEmb.Checked

theorem shape_valid (c : Call) (accepted : check c=true) : c.shape.Valid := by
  simp only [check,Bool.and_eq_true] at accepted
  have h := accepted.1.1
  simp only [shapeCheck,decide_eq_true_eq] at h
  exact ⟨h.1,h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2.1,h.2.2.2.2.2.2⟩

theorem extent_bound (c : Call) (accepted : check c=true) (i : Fin 4) :
    extent c.shape i≤(2^63-1)/4 := by
  simp only [check,Bool.and_eq_true] at accepted
  have h := accepted.1.2
  simp only [metadataCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  exact (h.2 i).2.2.2.2.2.2.2.2.2.2

theorem admitted (c : Call) (accepted : check c=true) : Host.Admitted c.metadata := by
  refine ⟨shape_valid c accepted,?_,?_,?_⟩
  · intro key
    have h := extent_bound c accepted (role key)
    cases key <;> simpa [Call.metadata,extent,expectedShape,role,Nat.mul_assoc,Shape.heads] using h
  · simpa [Call.metadata,extent,expectedShape] using extent_bound c accepted 2
  · simpa [Call.metadata,extent,expectedShape] using extent_bound c accepted 3

theorem active_nonempty (s : Shape) (w : Work) (active : s.Active w) : Host.empty s=false := by
  have := active.1
  have := active.2.2
  simp only [Shape.pairs] at this
  simp only [Host.empty,decide_eq_false_iff_not,not_or]
  omega

theorem launch_checked (c : Call) (accepted : check c=true) (w : Work) (active : c.shape.Active w) :
    launchCheck c=true := by
  simp only [check,Bool.and_eq_true] at accepted
  have h := accepted.2
  simpa [active_nonempty c.shape w active] using h

theorem launch_tensor (c : Call) (accepted : check c=true) (w : Work) (active : c.shape.Active w)
    (i : Fin 4) :
    (c.tensor i).strides=expectedStrides c.shape i ∧
    (c.tensor i).base=(c.tensor i).allocationBase ∧ (c.tensor i).base%4=0 ∧
    4*extent c.shape i≤(c.tensor i).allocationBytes ∧
    (c.tensor i).base+(c.tensor i).allocationBytes≤2^64 ∧
    4*lastOffset c.shape i+4<2^63 ∧ (c.tensor i).base+4*lastOffset c.shape i+4≤2^64 := by
  have h := launch_checked c accepted w active
  simp only [launchCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  exact h.1.1 i

theorem backing_separate (c : Call) (accepted : check c=true) (w : Work) (active : c.shape.Active w)
    (i j : Fin 4) (different : i≠j) : separate (c.tensor i) (c.tensor j) := by
  have h := launch_checked c accepted w active
  simp only [launchCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  exact h.1.2 i j different

theorem data_contained (c : Call) (accepted : check c=true) (w : Work)
    (active : c.shape.Active w) (odd : Bool) :
    4*c.shape.cell w odd+4≤(c.tensor (role w.key)).allocationBytes := by
  have bound := Host.data_offset_bound c.shape (shape_valid c accepted) w active odd
  have fits := (launch_tensor c accepted w active (role w.key)).2.2.2.1
  have extentEq : extent c.shape (role w.key)=c.shape.tokens*c.shape.heads w.key*c.shape.width := by
    cases w.key <;> simp [extent,role,expectedShape,Shape.heads,Nat.mul_assoc]
  rw [extentEq] at fits
  omega

/-- Every padded data expression is bounded, including masked lanes. The
kernel widens program IDs and arange lanes before multiplication. -/
theorem padded_data_bound (s : Shape) (key : Bool) (n h d : Nat)
    (hn : n<16*s.seqBlocks) (hh : h<4*s.headBlocks) (hd : d<2*s.pairBlock) :
    s.address key n h d≤lastOffset s (role key) := by
  have hn' : n≤16*s.seqBlocks-1 := by omega
  have hh' : h≤4*s.headBlocks-1 := by omega
  have hd' : d≤2*s.pairBlock-1 := by omega
  have h1 := Nat.mul_le_mul_right (s.heads key*s.width) hn'
  have h2 := Nat.mul_le_mul_right s.width hh'
  cases key <;> simp [Shape.address,Shape.heads,role,lastOffset,Nat.add_mul,Nat.mul_assoc] at * <;> omega

theorem padded_coefficient_bound (s : Shape) (sine : Bool) (n d : Nat)
    (hn : n<16*s.seqBlocks) (hd : d<2*s.pairBlock) :
    s.coefficient sine n d≤lastOffset s (coefficientRole sine) := by
  have hn' : n≤16*s.seqBlocks-1 := by omega
  have hd' : d≤2*s.pairBlock-1 := by omega
  have h := Nat.mul_le_mul_right (if sine then s.sinWidth else s.cosWidth) hn'
  cases sine <;> simp [Shape.coefficient,coefficientRole,lastOffset] at * <;> omega

#axiomsClean admitted
#axiomsClean data_contained
#axiomsClean padded_data_bound
#axiomsClean padded_coefficient_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RotaryEmb.Checked
