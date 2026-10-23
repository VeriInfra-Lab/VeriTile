import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityBase

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality.BackwardNoninner
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPBackwardNoninner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α : Type} [Inhabited α]

theorem load_equal (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row col rows N K B C cur : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) :
    load A s X row col rows N K B C cur=load A t X row col rows N K B C cur := by
  funext i
  unfold load
  congr 1
  split
  · apply h X hx
    have ha : cur+i.1.val<N ∧ col*C+i.2.1.val<K := by
      simpa only [active,Bool.and_eq_true,decide_eq_true_eq] using ‹active row col rows N K cur i=Bool.true›
    exact RealAddress.non_inner_bound hr ha.1 ha.2
  · rfl

theorem accumulate_equal (A : Algebra α) (regions : List RegionName) (P G : RegionName)
    (row col rows N K B C cur : Nat) (hr : row<rows) (hp : P∈regions) (hg : G∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) (v : Values α .real [B,C]) :
    accumulate A s P G row col rows N K B C cur v=accumulate A t P G row col rows N K B C cur v := by
  have sameP := load_equal A regions P row col rows N K B C cur hr hp s t h
  have sameG := load_equal A regions G row col rows N K B C cur hr hg s t h
  funext i
  simp only [accumulate,sameP,sameG]

theorem emit_agree (A : Algebra α) (regions : List RegionName) (P G Y : RegionName)
    (row col rows N K B C cur : Nat) (hr : row<rows) (hp : P∈regions) (hg : G∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) (v : Values α .real [C]) :
    Agree regions (rows*N*K) (emit A P G Y row col rows N K B C v cur s)
      (emit A P G Y row col rows N K B C v cur t) := by
  have sameP := load_equal A regions P row col rows N K B C cur hr hp s t h
  have sameG := load_equal A regions G row col rows N K B C cur hr hg s t h
  apply scatter_agree regions (rows*N*K) s t h Y
  funext i
  simp only [storedValue,sameP,sameG]

theorem graph_agree (A : Algebra α) (regions : List RegionName) (P G Y : RegionName)
    (row col rows N K B C : Nat) (hr : row<rows) (hp : P∈regions) (hg : G∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) :
    Agree regions (rows*N*K) (graph A s P G Y row col rows N K B C) (graph A t P G Y row col rows N K B C) := by
  have equal : (walk (accumulate A s P G row col rows N K B C) N B 0 (fun _ => literal A 0)).2=
      (walk (accumulate A t P G row col rows N K B C) N B 0 (fun _ => literal A 0)).2 := by
    apply FPGraphMemory.walk_rel _ _ Eq
    · intro cur a b hab
      subst b
      exact accumulate_equal A regions P G row col rows N K B C cur hr hp hg s t h a
    · rfl
  unfold graph
  rw [equal]
  apply FPGraphMemory.walk_rel _ _ (Agree regions (rows*N*K))
  · intro cur a b hab
    exact emit_agree A regions P G Y row col rows N K B C cur hr hp hg a b hab _
  · exact h

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality.BackwardNoninner
