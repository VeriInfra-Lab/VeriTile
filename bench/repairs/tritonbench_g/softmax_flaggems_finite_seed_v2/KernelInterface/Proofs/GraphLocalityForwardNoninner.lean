import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityBase

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality.ForwardNoninner
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPForwardNoninner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α : Type} [Inhabited α]

theorem load_equal (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row col rows N K B C cur : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) :
    load A s X row col N K B C cur=load A t X row col N K B C cur := by
  funext i
  unfold load
  congr 1
  split
  · apply h X hx
    have ha : cur+i.1.val<N ∧ col*C+i.2.1.val<K := by
      simpa only [active,Bool.and_eq_true,decide_eq_true_eq] using ‹active col N K cur i=Bool.true›
    exact RealAddress.non_inner_bound hr ha.1 ha.2
  · rfl

theorem accumulate_equal (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row col rows N K B C cur : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) (v : Acc α B C) :
    accumulate A s X row col N K B C cur v=accumulate A t X row col N K B C cur v := by
  have same := load_equal A regions X row col rows N K B C cur hr hx s t h
  simp only [accumulate,same]

theorem emit_agree (A : Algebra α) (regions : List RegionName) (X Y : RegionName)
    (row col rows N K B C cur : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) (mx z : Values α .real [C]) :
    Agree regions (rows*N*K) (emit A X Y row col N K B C mx z cur s)
      (emit A X Y row col N K B C mx z cur t) := by
  have same := load_equal A regions X row col rows N K B C (previous N B-cur) hr hx s t h
  apply scatter_agree regions (rows*N*K) s t h Y
  funext i
  simp only [storedValue,same]

theorem graph_agree (A : Algebra α) (regions : List RegionName) (X Y : RegionName)
    (row col rows N K B C : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) :
    Agree regions (rows*N*K) (graph A s X Y row col N K B C) (graph A t X Y row col N K B C) := by
  let seed : Acc α B C := ((fun _ => literal A (-340282346638528859811704183484516925440.0)),fun _ => literal A 0)
  have equal : (walk (accumulate A s X row col N K B C) N B 0 seed).2=
      (walk (accumulate A t X row col N K B C) N B 0 seed).2 := by
    apply FPGraphMemory.walk_rel _ _ Eq
    · intro cur a b hab
      subst b
      exact accumulate_equal A regions X row col rows N K B C cur hr hx s t h a
    · rfl
  unfold graph
  change Agree regions (rows*N*K)
    (walk (emit A X Y row col N K B C (maxReduced A B C _) (sumReduced A B C _)) N B 0 s).2
    (walk (emit A X Y row col N K B C (maxReduced A B C _) (sumReduced A B C _)) N B 0 t).2
  rw [equal]
  apply FPGraphMemory.walk_rel _ _ (Agree regions (rows*N*K))
  · intro cur a b hab
    exact emit_agree A regions X Y row col rows N K B C cur hr hx a b hab _ _
  · exact h

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality.ForwardNoninner
