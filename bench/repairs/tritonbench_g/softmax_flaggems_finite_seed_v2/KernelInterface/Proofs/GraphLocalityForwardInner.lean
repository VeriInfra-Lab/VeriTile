import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityBase

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality.ForwardInner
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPForwardInner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α : Type} [Inhabited α]

theorem load_equal (A : Algebra α) (regions : List RegionName) (X : RegionName) (masked : Bool)
    (row rows N B cur : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t)
    (ha : ∀ i : Fin B,active masked N cur (i,PUnit.unit) → cur+i.val<N) :
    load A s X masked row N B cur=load A t X masked row N B cur := by
  funext i
  unfold load
  congr 1
  split
  · apply h X hx
    apply inner_bound hr
    exact ha i.1 (by simpa only [Prod.mk.eta] using ‹active masked N cur i=Bool.true›)
  · rfl

theorem accumulate_equal (A : Algebra α) (regions : List RegionName) (X : RegionName) (masked : Bool)
    (row rows N B cur : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t)
    (ha : ∀ i : Fin B,active masked N cur (i,PUnit.unit) → cur+i.val<N) (v : Acc α B) :
    accumulate A s X masked row N B cur v=accumulate A t X masked row N B cur v := by
  have same := load_equal A regions X masked row rows N B cur hr hx s t h ha
  simp only [accumulate,same]

theorem emit_agree (A : Algebra α) (regions : List RegionName) (X Y : RegionName) (masked : Bool)
    (row rows N B cur : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t)
    (ha : ∀ i : Fin B,active masked N (previous N B-cur) (i,PUnit.unit) → previous N B-cur+i.val<N)
    (mx z : α) :
    Agree regions (rows*N) (emit A X Y masked row N B mx z cur s)
      (emit A X Y masked row N B mx z cur t) := by
  have same := load_equal A regions X masked row rows N B (previous N B-cur) hr hx s t h ha
  apply scatter_agree regions (rows*N) s t h Y
  funext i
  simp only [storedValue,same]

theorem graph_agree (A : Algebra α) (regions : List RegionName) (X Y : RegionName)
    (row rows N B : Nat) (hB : 0<B) (hBN : B<N) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t) :
    Agree regions (rows*N) (graph A s X Y row N B) (graph A t X Y row N B) := by
  have hN : 0<N := lt_trans hB hBN
  let seed : Acc α B := ((fun _ => literal A (-340282346638528859811704183484516925440.0)),fun _ => literal A 0)
  let firstS := (walk (accumulate A s X Bool.false row N B) (previous N B) B 0 seed).2
  let firstT := (walk (accumulate A t X Bool.false row N B) (previous N B) B 0 seed).2
  have equal₀ : firstS=firstT := by
    apply Locality.walk_rel _ _ Eq (fun cur => cur%B=0) (previous N B) B
    · intro cur hc
      simpa [Nat.add_mod] using hc
    · intro cur a b hc hi hab
      subst b
      exact accumulate_equal A regions X Bool.false row rows N B cur hr hx s t h
        (fun i _ => SourceSafety.ForwardInner.first_bound N B cur hN hB hi hc i) a
    · exact Nat.zero_mod _
    · rfl
  let valS := (walk (accumulate A s X Bool.true row N B) N B (previous N B) firstS).2
  let valT := (walk (accumulate A t X Bool.true row N B) N B (previous N B) firstT).2
  have equal₁ : valS=valT := by
    apply FPGraphMemory.walk_rel _ _ Eq
    · intro cur a b hab
      subst b
      exact accumulate_equal A regions X Bool.true row rows N B cur hr hx s t h
        (fun i hi => by simpa only [active,Bool.not_true,Bool.false_or,decide_eq_true_eq] using hi) a
    · exact equal₀
  let mx := maxReduced A B valT
  let z := sumReduced A B valT
  have out₀ : Agree regions (rows*N)
      (walk (emit A X Y Bool.true row N B mx z) B B 0 s).2
      (walk (emit A X Y Bool.true row N B mx z) B B 0 t).2 := by
    apply FPGraphMemory.walk_rel _ _ (Agree regions (rows*N))
    · intro cur a b hab
      exact emit_agree A regions X Y Bool.true row rows N B cur hr hx a b hab
        (fun i hi => by simpa only [active,Bool.not_true,Bool.false_or,decide_eq_true_eq] using hi) mx z
    · exact h
  unfold graph
  change Agree regions (rows*N)
    (walk (emit A X Y Bool.false row N B (maxReduced A B valS) (sumReduced A B valS)) N B B
      (walk (emit A X Y Bool.true row N B (maxReduced A B valS) (sumReduced A B valS)) B B 0 s).2).2
    (walk (emit A X Y Bool.false row N B mx z) N B B
      (walk (emit A X Y Bool.true row N B mx z) B B 0 t).2).2
  rw [equal₁]
  apply Locality.walk_rel _ _ (Agree regions (rows*N)) (fun cur => B≤cur) N B
  · intro cur hc
    exact Nat.le_trans hc (Nat.le_add_right cur B)
  · intro cur a b _ hi hab
    exact emit_agree A regions X Y Bool.false row rows N B cur hr hx a b hab
      (fun i _ => SourceSafety.ForwardInner.reverse_bound N B cur hN hB hBN hi i) _ _
  · exact le_refl _
  · exact out₀

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality.ForwardInner
