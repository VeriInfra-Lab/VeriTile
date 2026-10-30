import bench.tritonbench_g.rotary_emb_nopad.Contract

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Checked

theorem shapes (c : Call) (indices : Indices) (accepted : check c indices=true) :
    0<c.request.qHeads ∧ 0<c.request.kHeads ∧ c.request.qHeads%c.request.kHeads=0 ∧
    c.request.tokens≤c.request.cosRows ∧ c.request.tokens≤c.request.sinRows ∧
    c.request.half≤c.request.cosWidth ∧ c.request.half≤c.request.sinWidth ∧
    (c.request.cached=true → 0<c.request.blockSize) := by
  simp only [check,Bool.and_eq_true] at accepted
  simpa only [shapeCheck,decide_eq_true_eq] using accepted.1.1

theorem tensor_checked (c : Call) (indices : Indices) (accepted : check c indices=true) : tensorCheck c=true := by
  simp only [check,Bool.and_eq_true] at accepted
  exact accepted.1.2

theorem active_nonempty (r : Request) (w : Work) (active : r.Active w) : empty r=false := by
  have := active.1
  have := active.2.2
  simp only [empty,decide_eq_false_iff_not,not_or]
  omega

theorem launch_checked (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) : launchCheck c indices=true := by
  simp only [check,Bool.and_eq_true] at accepted
  have h := accepted.2
  simpa [active_nonempty c.request w active] using h

theorem metadata_checked (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) : metadataCheck c.request indices=true := by
  have h := launch_checked c indices accepted w active
  simp only [launchCheck,Bool.and_eq_true] at h
  exact h.1.1.1

theorem current_device_checked (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) : c.currentDevice=c.request.device := by
  have h := launch_checked c indices accepted w active
  simp only [launchCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  exact h.1.1.2

theorem groups_positive (c : Call) (indices : Indices) (accepted : check c indices=true) :
    0<c.request.groups ∧ c.request.kHeads*c.request.groups=c.request.qHeads := by
  have h := shapes c indices accepted
  have eq := Nat.mod_add_div c.request.qHeads c.request.kHeads
  rw [h.2.2.1,zero_add] at eq
  change 0<c.request.qHeads/c.request.kHeads ∧ c.request.kHeads*(c.request.qHeads/c.request.kHeads)=c.request.qHeads
  exact ⟨by nlinarith [h.1],eq⟩

theorem slots (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) (cached : c.request.cached=true) (t : Nat) (ht : t<c.request.tokens) :
    c.request.block indices t<c.request.blocks ∧ c.request.slot indices t<c.request.blockSize := by
  have h := metadata_checked c indices accepted w active
  simp only [metadataCheck,cached,↓reduceIte,Bool.and_eq_true,decide_eq_true_eq] at h
  exact ⟨(h.1 ⟨t,ht⟩).2.2.2.2,Nat.mod_lt _ ((shapes c indices accepted).2.2.2.2.2.2.2 cached)⟩

theorem slots_unique (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) (cached : c.request.cached=true)
    (a b : Nat) (ha : a<c.request.tokens) (hb : b<c.request.tokens)
    (same : (c.request.block indices a,c.request.slot indices a)=(c.request.block indices b,c.request.slot indices b)) : a=b := by
  have h := metadata_checked c indices accepted w active
  simp only [metadataCheck,cached,↓reduceIte,Bool.and_eq_true,decide_eq_true_eq] at h
  by_contra ne
  exact h.2 ⟨a,ha⟩ ⟨b,hb⟩ (by intro eq; exact ne (congrArg Fin.val eq)) same

#axiomsClean current_device_checked
#axiomsClean groups_positive
#axiomsClean slots
#axiomsClean slots_unique
end VeriTile.Bench.Repairs.RotaryEmbNopad.Checked
