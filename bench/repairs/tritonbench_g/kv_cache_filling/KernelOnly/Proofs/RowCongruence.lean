import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.IO

namespace VeriTile.Bench.Repairs.KvCacheFilling.Interface
open VeriTile.Triton FP.Structural QuantProfile
noncomputable section

theorem pad_congr {α : Type} {w n : Nat} (pos : 0 < w) {x y : Nat → α}
    (h : ∀ i, i < w → x i = y i) : pad w n x = pad w n y := by
  funext i
  apply h
  split <;> omega

theorem row8_congr {α : Type} (a : Ops α) {w n : Nat} (pos : 0 < w)
    {x y : Nat → α} (h : ∀ i, i < w → x i = y i) :
    (∀ i, i < w → (row8 a w n x).values i = (row8 a w n y).values i) ∧
    (row8 a w n x).scale = (row8 a w n y).scale ∧
    (row8 a w n x).zero = (row8 a w n y).zero := by
  have pads : pad w n x = pad w n y := pad_congr pos h
  simp only [row8, pads]
  exact ⟨fun i hi => congrArg _ (h i hi),True.intro,True.intro⟩

theorem row4_congr {α : Type} (a : Ops α) {w n : Nat} (pos : 0 < w)
    {x y : Nat → α} (h : ∀ i, i < 2*w → x i = y i) :
    (∀ i, i < w → (row4 a w n x).values i = (row4 a w n y).values i) ∧
    (row4 a w n x).scale = (row4 a w n y).scale ∧
    (row4 a w n x).zero = (row4 a w n y).zero := by
  have left : pad w n x = pad w n y := pad_congr pos (fun i hi => h i (by omega))
  have right : pad w n (fun i => x (w+i)) = pad w n (fun i => y (w+i)) :=
    pad_congr pos (fun i hi => h (w+i) (by omega))
  simp only [row4, left, right]
  refine ⟨?_,True.intro,True.intro⟩
  intro i hi
  rw [h i (by omega),h (w+i) (by omega)]

theorem row_congr {α : Type} (a : Ops α) (policy : Policy) {w n : Nat} (pos : 0 < w)
    {x y : Nat → α} (h : ∀ i, i < (if policy = .int4 then 2*w else w) → x i = y i) :
    (∀ i, i < w → (SharedTarget.row a policy w n x).value i =
      (SharedTarget.row a policy w n y).value i) ∧
    (SharedTarget.row a policy w n x).scale = (SharedTarget.row a policy w n y).scale ∧
    (SharedTarget.row a policy w n x).zero = (SharedTarget.row a policy w n y).zero := by
  cases policy
  · exact ⟨fun i hi => congrArg Cell.fp32 (h i (by simpa using hi)),rfl,rfl⟩
  · have hc := row4_congr a (n := n) pos (by simpa using h)
    exact ⟨fun i hi => congrArg Cell.uint8 (hc.1 i hi),hc.2⟩
  · have hc := row8_congr a (n := n) pos (by simpa using h)
    exact ⟨fun i hi => congrArg Cell.uint8 (hc.1 i hi),hc.2⟩

theorem result_congr {α : Type} [Inhabited α] (a : Ops α) (r : Request) (d : Domain r)
    {kx ky vx vy : Nat → α} (hk : ∀ i, i < r.dk → kx i = ky i)
    (hv : ∀ i, i < r.dv → vx i = vy i) :
    result r (SharedTarget.value a r kx vx) = result r (SharedTarget.value a r ky vy) := by
  have pos := packed_widths_positive r d
  have ks : r.dk = if r.policy = .int4 then 2*r.ck else r.ck := by
    simpa [Request.factor,ite_mul] using d.key_shape
  have vs : r.dv = if r.policy = .int4 then 2*r.cv else r.cv := by
    simpa [Request.factor,ite_mul] using d.value_shape
  have key := row_congr a r.policy (n := r.ck.nextPowerOfTwo) pos.1
    (fun i hi => hk i (by rwa [ks]))
  have val := row_congr a r.policy (n := r.cv.nextPowerOfTwo) pos.2
    (fun i hi => hv i (by rwa [vs]))
  funext o k
  fin_cases o
  · exact congrArg (cacheValue r.policy) (key.1 k.1.val k.1.isLt)
  · exact congrArg (cacheValue r.policy) (val.1 k.1.val k.1.isLt)
  · change (if k.1.val = 0 then _ else _) = (if k.1.val = 0 then _ else _)
    rw [key.2.1,key.2.2]
  · change (if k.1.val = 0 then _ else _) = (if k.1.val = 0 then _ else _)
    rw [val.2.1,val.2.2]

end
end VeriTile.Bench.Repairs.KvCacheFilling.Interface
