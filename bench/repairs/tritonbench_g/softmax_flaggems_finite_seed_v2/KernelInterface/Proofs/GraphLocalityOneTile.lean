import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityBase

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed FPReference
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward_inner_load (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row rows N B : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t) :
    forwardInnerLoad A s X row N B=forwardInnerLoad A t X row N B := by
  funext i
  unfold forwardInnerLoad
  congr 1
  split
  · apply h X hx
    apply inner_bound hr
    simpa only [innerActive,decide_eq_true_eq] using ‹innerActive N i=Bool.true›
  · rfl

theorem forward_noninner_load (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row col rows N K B C : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) :
    forwardNoninnerLoad A s X row col N K B C=forwardNoninnerLoad A t X row col N K B C := by
  funext i
  unfold forwardNoninnerLoad
  congr 1
  split
  · apply h X hx
    have ha : i.1.val<N ∧ col*C+i.2.1.val<K := by
      simpa only [noninnerActive,Bool.and_eq_true,decide_eq_true_eq] using ‹noninnerActive col N K i=Bool.true›
    exact RealAddress.non_inner_bound hr ha.1 ha.2
  · rfl

theorem backward_inner_load (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row rows N B C : Nat) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t) :
    backwardInnerLoad A s X row rows N B C=backwardInnerLoad A t X row rows N B C := by
  funext i
  unfold backwardInnerLoad
  congr 1
  split
  · apply h X hx
    have ha : row*C+i.1.val<rows ∧ i.2.1.val<N := by
      simpa only [backwardInnerActive,Bool.and_eq_true,decide_eq_true_eq] using ‹backwardInnerActive row rows N i=Bool.true›
    exact inner_bound ha.1 ha.2
  · rfl

theorem backward_noninner_load (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row col rows N K B C : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) :
    backwardNoninnerLoad A s X row col N K B C=backwardNoninnerLoad A t X row col N K B C := by
  funext i
  unfold backwardNoninnerLoad
  congr 1
  split
  · apply h X hx
    have ha : i.1.val<N ∧ col*C+i.2.1.val<K := by
      simpa only [noninnerActive,Bool.and_eq_true,decide_eq_true_eq] using ‹noninnerActive col N K i=Bool.true›
    exact RealAddress.non_inner_bound hr ha.1 ha.2
  · rfl

theorem forward_inner_value (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row rows N B : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t) :
    forwardInnerValue A s X row N B=forwardInnerValue A t X row N B := by
  have same := forward_inner_load A regions X row rows N B hr hx s t h
  have sameE : forwardInnerExp A s X row N B=forwardInnerExp A t X row N B := by
    funext j
    simp only [forwardInnerExp,same]
  funext i
  simp only [forwardInnerValue,sameE]

theorem forward_noninner_value (A : Algebra α) (regions : List RegionName) (X : RegionName)
    (row col rows N K B C : Nat) (hr : row<rows) (hx : X∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) :
    forwardNoninnerValue A s X row col N K B C=forwardNoninnerValue A t X row col N K B C := by
  have same := forward_noninner_load A regions X row col rows N K B C hr hx s t h
  have sameE : forwardNoninnerExp A s X row col N K B C=forwardNoninnerExp A t X row col N K B C := by
    funext j
    simp only [forwardNoninnerExp,same]
  funext i
  simp only [forwardNoninnerValue,sameE]

theorem backward_inner_value (A : Algebra α) (regions : List RegionName) (P G : RegionName)
    (row rows N B C : Nat) (hp : P∈regions) (hg : G∈regions) (s t : State α)
    (h : Agree regions (rows*N) s t) :
    backwardInnerValue A s P G row rows N B C=backwardInnerValue A t P G row rows N B C := by
  have sameP := backward_inner_load A regions P row rows N B C hp s t h
  have sameG := backward_inner_load A regions G row rows N B C hg s t h
  funext i
  simp only [backwardInnerValue,backwardInnerScale,sameP,sameG]

theorem backward_noninner_value (A : Algebra α) (regions : List RegionName) (P G : RegionName)
    (row col rows N K B C : Nat) (hr : row<rows) (hp : P∈regions) (hg : G∈regions) (s t : State α)
    (h : Agree regions (rows*N*K) s t) :
    backwardNoninnerValue A s P G row col N K B C=backwardNoninnerValue A t P G row col N K B C := by
  have sameP := backward_noninner_load A regions P row col rows N K B C hr hp s t h
  have sameG := backward_noninner_load A regions G row col rows N K B C hr hg s t h
  funext i
  simp only [backwardNoninnerValue,backwardNoninnerScale,sameP,sameG]

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality
