import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.FillCode
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

theorem Allocation.ne (a : Allocation n caller) (i j : Fin n) (h : i≠j) :
    a.storage i≠a.storage j := fun he => h (a.distinct he)

theorem Allocation.input_fresh (a : Allocation n caller) (b : InputBuffers)
    (inputs : ∀ id,id∈b.storageIds → id∈caller) (i : Fin n) : a.storage i∉b.storageIds := by
  intro hi
  exact a.fresh i (inputs _ hi)

theorem partialPhase_frame (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (seed final : Bool) (kind : PartialKind) (storage count : Nat) (m : Memory α) (q : Address)
    (hq : q.1≠storage) : partialPhase a s p b seed final kind storage count m q=m q :=
  fillRange_frame _ _ _ _ _ hq

theorem partialPhase_observations (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (seed final : Bool) (kind : PartialKind) (storage count : Nat) (m : Memory α)
    (fresh : storage∉b.storageIds) :
    b.observe s (partialPhase a s p b seed final kind storage count m)=b.observe s m :=
  fillRange_observations b s storage count _ m fresh

theorem partialO_at (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α)
    (seed final : Bool) (split h t v : Nat) (hh : h<s.BH) (ht : t<s.T) (hv : v<s.DV) :
    partialValue a s p x seed final .output (partialOffset s s.DV split h t v)=
      partialO a s p x seed h t split v := by
  have he := flat_partial s s.DV split h t v hh ht hv
  simp [partialValue,PartialKind.width,he.1,he.2.1,he.2.2.1,he.2.2.2]

theorem partialQ_at (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α)
    (seed final : Bool) (split h t k : Nat) (hh : h<s.BH) (ht : t<s.T) (hk : k<s.DK) :
    partialValue a s p x seed final .dq (partialOffset s s.DK split h t k)=
      partialQ a s p x seed h t split k := by
  have he := flat_partial s s.DK split h t k hh ht hk
  simp [partialValue,PartialKind.width,he.1,he.2.1,he.2.2.1,he.2.2.2]

theorem partialK_at (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α)
    (seed final : Bool) (split h t k : Nat) (hh : h<s.BH) (ht : t<s.T) (hk : k<s.DK) :
    partialValue a s p x seed final .dk (partialOffset s s.DK split h t k)=
      partialK a s p x final h t split k := by
  have he := flat_partial s s.DK split h t k hh ht hk
  simp [partialValue,PartialKind.width,he.1,he.2.1,he.2.2.1,he.2.2.2]

theorem partialV_at (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α)
    (seed final : Bool) (split h t v : Nat) (hh : h<s.BH) (ht : t<s.T) (hv : v<s.DV) :
    partialValue a s p x seed final .dv (partialOffset s s.DV split h t v)=
      partialV a s p x final h t split v := by
  have he := flat_partial s s.DV split h t v hh ht hv
  simp [partialValue,PartialKind.width,he.1,he.2.1,he.2.2.1,he.2.2.2]

theorem partialState_at (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α)
    (seed final : Bool) (h k v : Nat) (hk : k<s.DK) (hv : v<s.DV) :
    partialValue a s p x seed final .finalState (stateOffset s h k v)=state a s x seed h k v s.T := by
  simp only [partialValue,stateOffset,pair_div s.DV _ v hv,pair_div s.DK h k hk,
    pair_mod s.DV _ v hv,pair_mod s.DK h k hk]

#axiomsClean Allocation.input_fresh
#axiomsClean partialO_at
#axiomsClean partialQ_at
#axiomsClean partialK_at
#axiomsClean partialV_at
#axiomsClean partialState_at
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
