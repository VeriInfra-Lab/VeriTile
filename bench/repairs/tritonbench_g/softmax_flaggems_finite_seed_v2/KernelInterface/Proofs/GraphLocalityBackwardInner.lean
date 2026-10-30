import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityBase

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality.BackwardInner
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPBackwardInner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α : Type} [Inhabited α]

theorem load_equal (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row rows N B C cur : Nat) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t) :
    load A s X row rows N B C cur=load A t X row rows N B C cur := by
  funext i
  unfold load
  congr 1
  split
  · apply h X hx
    have ha : row*C+i.1.val<rows ∧ cur+i.2.1.val<N := by
      simpa only [active,Bool.and_eq_true,decide_eq_true_eq] using ‹active row rows N cur i=Bool.true›
    exact inner_bound ha.1 ha.2
  · rfl

theorem accumulate_equal (A : Algebra α) (regions : List RegionName) (P G : RegionName)
    (row rows N B C cur : Nat) (hp : P∈regions) (hg : G∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t) (v : Values α .real [C,B]) :
    accumulate A s P G row rows N B C cur v=accumulate A t P G row rows N B C cur v := by
  have sameP := load_equal A regions P row rows N B C cur hp s t h
  have sameG := load_equal A regions G row rows N B C cur hg s t h
  funext i
  simp only [accumulate,sameP,sameG]

theorem emit_agree (A : Algebra α) (regions : List RegionName) (P G Y : RegionName)
    (row rows N B C cur : Nat) (hp : P∈regions) (hg : G∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t) (v : Values α .real [C]) :
    Agree regions (rows*N) (emit A P G Y row rows N B C v cur s)
      (emit A P G Y row rows N B C v cur t) := by
  have sameP := load_equal A regions P row rows N B C cur hp s t h
  have sameG := load_equal A regions G row rows N B C cur hg s t h
  apply scatter_agree regions (rows*N) s t h Y
  funext i
  simp only [storedValue,sameP,sameG]

theorem graph_agree (A : Algebra α) (regions : List RegionName) (P G Y : RegionName)
    (row rows N B C : Nat) (hp : P∈regions) (hg : G∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t) :
    Agree regions (rows*N) (graph A s P G Y row rows N B C) (graph A t P G Y row rows N B C) := by
  have equal : (walk (accumulate A s P G row rows N B C) N B 0 (fun _ => literal A 0)).2=
      (walk (accumulate A t P G row rows N B C) N B 0 (fun _ => literal A 0)).2 := by
    apply FPGraphMemory.walk_rel _ _ Eq
    · intro cur a b hab
      subst b
      exact accumulate_equal A regions P G row rows N B C cur hp hg s t h a
    · rfl
  unfold graph
  rw [equal]
  apply FPGraphMemory.walk_rel _ _ (Agree regions (rows*N))
  · intro cur a b hab
    exact emit_agree A regions P G Y row rows N B C cur hp hg a b hab _
  · exact h

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality.BackwardInner
