import bench.tritonbench_g.fused_recurrent_retention.Proofs.Reference
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
open scoped BigOperators

/- Fixed, numerical-free plans selected by the concrete compiler/runtime
profile. No plan receives input words, a carry, or an expected result. -/
def partialO (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α) (seed : Bool)
    (h t split v : Nat) : α :=
  p.output.eval a (fun j => state a s x seed h (split*s.BK+j) v (t+1))
    (fun j => a.mul (keyLoad a s x.q h t (split*s.BK+j)) (a.scale s.DK))
def partialQ (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α) (seed : Bool)
    (h t split k : Nat) : α :=
  a.mul (p.dq.eval a (fun j => state a s x seed h k (split*s.BV+j) (t+1))
    (fun j => valueLoad a s x.do_ h t (split*s.BV+j))) (a.scale s.DK)
def partialK (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α) (final : Bool)
    (h t split k : Nat) : α :=
  p.dk.eval a (fun j => adjoint a s x final h t k (split*s.BV+j))
    (fun j => valueLoad a s x.v h t (split*s.BV+j))
def partialV (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α) (final : Bool)
    (h t split v : Nat) : α :=
  p.dv.eval a (fun j => adjoint a s x final h t (split*s.BK+j) v)
    (fun j => keyLoad a s x.k h t (split*s.BK+j))

def resultO (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α) (seed : Bool) (h t v : Nat) : α :=
  (p.sumO (rowAddress s s.DV h t v)).eval a (fun split => partialO a s p x seed h t split v)
def resultQ (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α) (seed : Bool) (h t k : Nat) : α :=
  (p.sumQ (rowAddress s s.DK h t k)).eval a (fun split => partialQ a s p x seed h t split k)
def resultK (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α) (final : Bool) (h t k : Nat) : α :=
  (p.sumK (rowAddress s s.DK h t k)).eval a (fun split => partialK a s p x final h t split k)
def resultV (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α) (final : Bool) (h t v : Nat) : α :=
  (p.sumV (rowAddress s s.DV h t v)).eval a (fun split => partialV a s p x final h t split v)

theorem partialO_real (s : Shape) (p : Plans) (hp : p.Valid s) (x : Inputs ℝ)
    (seed : Bool) (h t split v : Nat) (hv : v<s.DV) :
    partialO realArithmetic s p x seed h t split v=
      ∑ j∈Finset.range s.BK,if split*s.BK+j<s.DK then
        mathState s x seed h (split*s.BK+j) v (t+1)*(x.q h t (split*s.BK+j)*realArithmetic.scale s.DK) else 0 := by
  unfold partialO
  rw [p.output.real_sum s.BK hp.output]
  apply Finset.sum_congr rfl
  intro j hj
  by_cases hk : split*s.BK+j<s.DK
  · rw [state_real s x seed h _ v _ hk hv]
    simp [keyLoad,hk,realArithmetic]
  · rw [state_real_inactive s x seed h _ v _ (Or.inl hk)]
    simp [hk]

theorem partialQ_real (s : Shape) (p : Plans) (hp : p.Valid s) (x : Inputs ℝ)
    (seed : Bool) (h t split k : Nat) (hk : k<s.DK) :
    partialQ realArithmetic s p x seed h t split k=
      (∑ j∈Finset.range s.BV,if split*s.BV+j<s.DV then
        mathState s x seed h k (split*s.BV+j) (t+1)*x.do_ h t (split*s.BV+j) else 0)*realArithmetic.scale s.DK := by
  unfold partialQ
  rw [p.dq.real_sum s.BV hp.dq]
  change (_ : ℝ)*_=(_ : ℝ)*_
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  by_cases hv : split*s.BV+j<s.DV
  · rw [state_real s x seed h k _ _ hk hv]
    simp [valueLoad,hv]
  · rw [state_real_inactive s x seed h k _ _ (Or.inr hv)]
    simp [hv]

theorem partialK_real (s : Shape) (p : Plans) (hp : p.Valid s) (x : Inputs ℝ)
    (final : Bool) (h t split k : Nat) (hk : k<s.DK) (ht : t<s.T) :
    partialK realArithmetic s p x final h t split k=
      ∑ j∈Finset.range s.BV,if split*s.BV+j<s.DV then
        mathAdjoint s x final h t k (split*s.BV+j)*x.v h t (split*s.BV+j) else 0 := by
  unfold partialK
  rw [p.dk.real_sum s.BV hp.dk]
  apply Finset.sum_congr rfl
  intro j hj
  by_cases hv : split*s.BV+j<s.DV
  · rw [adjoint_real s x final h t k _ hk hv ht]
    simp [valueLoad,hv]
  · rw [adjoint_real_inactive s x final h t k _ (Or.inr hv)]
    simp [hv]

theorem partialV_real (s : Shape) (p : Plans) (hp : p.Valid s) (x : Inputs ℝ)
    (final : Bool) (h t split v : Nat) (hv : v<s.DV) (ht : t<s.T) :
    partialV realArithmetic s p x final h t split v=
      ∑ j∈Finset.range s.BK,if split*s.BK+j<s.DK then
        mathAdjoint s x final h t (split*s.BK+j) v*x.k h t (split*s.BK+j) else 0 := by
  unfold partialV
  rw [p.dv.real_sum s.BK hp.dv]
  apply Finset.sum_congr rfl
  intro j hj
  by_cases hk : split*s.BK+j<s.DK
  · rw [adjoint_real s x final h t _ v hk hv ht]
    simp [keyLoad,hk]
  · rw [adjoint_real_inactive s x final h t _ v (Or.inl hk)]
    simp [hk]

theorem resultO_real (s : Shape) (domain : s.Admitted) (p : Plans) (hp : p.Valid s)
    (x : Inputs ℝ) (seed : Bool) (h t v : Nat) (hv : v<s.DV) :
    resultO realArithmetic s p x seed h t v=mathOutput s x seed h t v := by
  unfold resultO
  rw [(p.sumO (rowAddress s s.DV h t v)).real_sum s.NK (hp.sumO _)]
  simp_rw [partialO_real s p hp x seed h t _ v hv]
  rw [show s.NK=tileCount s.DK s.BK from rfl]
  rw [sum_tiles s.DK s.BK domain.1 (block_positive s domain).1
    (fun k => mathState s x seed h k v (t+1)*(x.q h t k*realArithmetic.scale s.DK))]
  unfold mathOutput
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k hk
  ring

theorem resultQ_real (s : Shape) (domain : s.Admitted) (p : Plans) (hp : p.Valid s)
    (x : Inputs ℝ) (seed : Bool) (h t k : Nat) (hk : k<s.DK) :
    resultQ realArithmetic s p x seed h t k=mathDQ s x seed h t k := by
  unfold resultQ
  rw [(p.sumQ (rowAddress s s.DK h t k)).real_sum s.NV (hp.sumQ _)]
  simp_rw [partialQ_real s p hp x seed h t _ k hk]
  rw [←Finset.sum_mul]
  rw [show s.NV=tileCount s.DV s.BV from rfl]
  rw [sum_tiles s.DV s.BV domain.2.1 (block_positive s domain).2
    (fun v => mathState s x seed h k v (t+1)*x.do_ h t v)]
  unfold mathDQ
  ring

theorem resultK_real (s : Shape) (domain : s.Admitted) (p : Plans) (hp : p.Valid s)
    (x : Inputs ℝ) (final : Bool) (h t k : Nat) (hk : k<s.DK) (ht : t<s.T) :
    resultK realArithmetic s p x final h t k=mathDK s x final h t k := by
  unfold resultK
  rw [(p.sumK (rowAddress s s.DK h t k)).real_sum s.NV (hp.sumK _)]
  simp_rw [partialK_real s p hp x final h t _ k hk ht]
  exact sum_tiles s.DV s.BV domain.2.1 (block_positive s domain).2
    (fun v => mathAdjoint s x final h t k v*x.v h t v)

theorem resultV_real (s : Shape) (domain : s.Admitted) (p : Plans) (hp : p.Valid s)
    (x : Inputs ℝ) (final : Bool) (h t v : Nat) (hv : v<s.DV) (ht : t<s.T) :
    resultV realArithmetic s p x final h t v=mathDV s x final h t v := by
  unfold resultV
  rw [(p.sumV (rowAddress s s.DV h t v)).real_sum s.NK (hp.sumV _)]
  simp_rw [partialV_real s p hp x final h t _ v hv ht]
  exact sum_tiles s.DK s.BK domain.1 (block_positive s domain).1
    (fun k => mathAdjoint s x final h t k v*x.k h t k)

#axiomsClean resultO_real
#axiomsClean resultQ_real
#axiomsClean resultK_real
#axiomsClean resultV_real
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
