import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Geometry
import bench.tritonbench_g.fused_recurrent_retention.Proofs.MaskedMath
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
open scoped BigOperators
namespace Legacy
abbrev prefixSum := VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.MaskedMath.decayedPrefix
abbrev suffix := VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.MaskedMath.decayedSuffix
end Legacy
open VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.MaskedMath

/- Logical observations of the original caller allocations. The last two
fields are optional inputs, never supplied intermediate states or outputs. -/
/- Scalar projection of the source's actual forward carry. -/
def state (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool) (h k v : Nat) : Nat → α :=
  fun n => foldN (fun t old => a.fma (decay a (h%s.H)) old
    (a.mul (keyLoad a s x.k h t k) (valueLoad a s x.v h t v))) n
    (if seed then a.add a.zero (stateLoad a s x.seed h k v) else a.zero)

def reverseTerm (a : Arithmetic α) (s : Shape) (x : Inputs α) (h t k v : Nat) : α :=
  a.mul (a.mul (keyLoad a s x.q h t k) (a.scale s.DK)) (valueLoad a s x.do_ h t v)

/- Carry after r reverse iterations, including the final decay even after
the last store. The initial carry is F, with no extra initial decay. -/
def reverseCarry (a : Arithmetic α) (s : Shape) (x : Inputs α) (hasFinal : Bool) (h k v : Nat) : Nat → α :=
  fun n => foldN (fun r old =>
    let t := s.T-1-r
    a.mul (a.fma (a.mul (keyLoad a s x.q h t k) (a.scale s.DK))
      (valueLoad a s x.do_ h t v) old) (decay a (h%s.H))) n
    (if hasFinal then stateLoad a s x.finalGrad h k v else a.zero)

def adjoint (a : Arithmetic α) (s : Shape) (x : Inputs α) (hasFinal : Bool) (h t k v : Nat) : α :=
  a.fma (a.mul (keyLoad a s x.q h t k) (a.scale s.DK))
    (valueLoad a s x.do_ h t v) (reverseCarry a s x hasFinal h k v (s.T-1-t))

/- Independent dense mathematical prefix and adjoint, with no tiles, plans,
mutable memory, native operations, or source execution in their definitions. -/
def mathState (s : Shape) (x : Inputs ℝ) (seed : Bool) (h k v n : Nat) : ℝ :=
  Legacy.prefixSum (decay realArithmetic (h%s.H)) (if seed then x.seed h k v else 0)
    (fun t => x.k h t k*x.v h t v) n
def mathAdjoint (s : Shape) (x : Inputs ℝ) (hasFinal : Bool) (h t k v : Nat) : ℝ :=
  Legacy.suffix (decay realArithmetic (h%s.H))
    (fun u => (x.q h u k*realArithmetic.scale s.DK)*x.do_ h u v) s.T t+
      (decay realArithmetic (h%s.H))^(s.T-1-t)*(if hasFinal then x.finalGrad h k v else 0)
def mathOutput (s : Shape) (x : Inputs ℝ) (seed : Bool) (h t v : Nat) : ℝ :=
  realArithmetic.scale s.DK*∑ k∈Finset.range s.DK,x.q h t k*mathState s x seed h k v (t+1)
def mathDQ (s : Shape) (x : Inputs ℝ) (seed : Bool) (h t k : Nat) : ℝ :=
  realArithmetic.scale s.DK*∑ v∈Finset.range s.DV,mathState s x seed h k v (t+1)*x.do_ h t v
def mathDK (s : Shape) (x : Inputs ℝ) (hasFinal : Bool) (h t k : Nat) : ℝ :=
  ∑ v∈Finset.range s.DV,mathAdjoint s x hasFinal h t k v*x.v h t v
def mathDV (s : Shape) (x : Inputs ℝ) (hasFinal : Bool) (h t v : Nat) : ℝ :=
  ∑ k∈Finset.range s.DK,mathAdjoint s x hasFinal h t k v*x.k h t k

theorem state_real (s : Shape) (x : Inputs ℝ) (seed : Bool) (h k v n : Nat)
    (hk : k<s.DK) (hv : v<s.DV) :
    state realArithmetic s x seed h k v n=mathState s x seed h k v n := by
  induction n with
  | zero =>
    simp [state,foldN,mathState,Legacy.prefixSum,prefix_zero,keyLoad,valueLoad,stateLoad,hk,hv,realArithmetic]
  | succ n ih =>
    change realArithmetic.fma (decay realArithmetic (h%s.H))
      (state realArithmetic s x seed h k v n)
      (realArithmetic.mul (keyLoad realArithmetic s x.k h n k) (valueLoad realArithmetic s x.v h n v))=_
    rw [ih]
    unfold mathState Legacy.prefixSum
    rw [prefix_succ]
    simp [keyLoad,valueLoad,hk,hv,realArithmetic]

theorem state_real_inactive (s : Shape) (x : Inputs ℝ) (seed : Bool) (h k v n : Nat)
    (inactive : ¬k<s.DK ∨ ¬v<s.DV) : state realArithmetic s x seed h k v n=0 := by
  induction n with
  | zero =>
    rcases inactive with hx|hx <;> simp [state,foldN,stateLoad,hx,realArithmetic]
  | succ n ih =>
    change realArithmetic.fma _ (state realArithmetic s x seed h k v n) _=0
    rw [ih]
    rcases inactive with hx|hx <;> simp [keyLoad,valueLoad,hx,realArithmetic]

theorem reverseCarry_real (s : Shape) (x : Inputs ℝ) (hasFinal : Bool) (h k v r : Nat)
    (hk : k<s.DK) (hv : v<s.DV) (hr : r ≤ s.T) :
    reverseCarry realArithmetic s x hasFinal h k v r=
      decay realArithmetic (h%s.H)*Legacy.suffix (decay realArithmetic (h%s.H))
        (fun u => (x.q h u k*realArithmetic.scale s.DK)*x.do_ h u v) s.T (s.T-r)+
        (decay realArithmetic (h%s.H))^r*(if hasFinal then x.finalGrad h k v else 0) := by
  induction r with
  | zero =>
    simp [reverseCarry,foldN,Legacy.suffix,suffix_end,stateLoad,hk,hv,realArithmetic]
  | succ r ih =>
    have hr' : r ≤ s.T := by omega
    have ht : s.T-1-r<s.T := by omega
    have he : s.T-1-r+1=s.T-r := by omega
    have hs := suffix_step (decay realArithmetic (h%s.H))
      (fun u => (x.q h u k*realArithmetic.scale s.DK)*x.do_ h u v) s.T (s.T-1-r) ht
    rw [he] at hs
    change realArithmetic.mul (realArithmetic.fma
      (realArithmetic.mul (keyLoad realArithmetic s x.q h (s.T-1-r) k) (realArithmetic.scale s.DK))
      (valueLoad realArithmetic s x.do_ h (s.T-1-r) v)
      (reverseCarry realArithmetic s x hasFinal h k v r)) (decay realArithmetic (h%s.H))=_
    rw [ih hr',show s.T-(r+1)=s.T-1-r by omega]
    change _=decay realArithmetic (h%s.H)*decayedSuffix _ _ _ _+_
    rw [hs,pow_succ]
    simp only [keyLoad,valueLoad,hk,hv,if_true,realArithmetic]
    ring

theorem adjoint_real (s : Shape) (x : Inputs ℝ) (hasFinal : Bool) (h t k v : Nat)
    (hk : k<s.DK) (hv : v<s.DV) (ht : t<s.T) :
    adjoint realArithmetic s x hasFinal h t k v=mathAdjoint s x hasFinal h t k v := by
  unfold adjoint
  rw [reverseCarry_real s x hasFinal h k v (s.T-1-t) hk hv (by omega)]
  have he : s.T-(s.T-1-t)=t+1 := by omega
  rw [he]
  unfold mathAdjoint Legacy.suffix
  rw [suffix_step _ _ _ _ ht]
  simp only [keyLoad,valueLoad,hk,hv,if_true,realArithmetic]
  ring

theorem reverseCarry_real_inactive (s : Shape) (x : Inputs ℝ) (hasFinal : Bool) (h k v r : Nat)
    (inactive : ¬k<s.DK ∨ ¬v<s.DV) : reverseCarry realArithmetic s x hasFinal h k v r=0 := by
  induction r with
  | zero =>
    rcases inactive with hx|hx <;> simp [reverseCarry,foldN,stateLoad,hx,realArithmetic]
  | succ r ih =>
    change realArithmetic.mul (realArithmetic.fma _ _ (reverseCarry realArithmetic s x hasFinal h k v r)) _=0
    rw [ih]
    rcases inactive with hx|hx <;> simp [keyLoad,valueLoad,hx,realArithmetic]

theorem adjoint_real_inactive (s : Shape) (x : Inputs ℝ) (hasFinal : Bool) (h t k v : Nat)
    (inactive : ¬k<s.DK ∨ ¬v<s.DV) : adjoint realArithmetic s x hasFinal h t k v=0 := by
  unfold adjoint
  rw [reverseCarry_real_inactive s x hasFinal h k v _ inactive]
  rcases inactive with hx|hx <;> simp [keyLoad,valueLoad,hx,realArithmetic]

#axiomsClean state_real
#axiomsClean state_real_inactive
#axiomsClean reverseCarry_real
#axiomsClean adjoint_real
#axiomsClean adjoint_real_inactive
#specNonCircular mathState avoiding [state,reverseCarry,adjoint]
#specNonCircular mathAdjoint avoiding [state,reverseCarry,adjoint]
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
