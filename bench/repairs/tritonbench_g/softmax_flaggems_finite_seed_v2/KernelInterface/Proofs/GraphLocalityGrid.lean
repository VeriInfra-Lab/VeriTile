import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityOneTile
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityBackwardInner
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityBackwardNoninner
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityForwardInner
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityForwardNoninner

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed FPReference
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward_block (A : Algebra α) (regions : List RegionName) (X Y : RegionName)
    (d : Dimensions) (c : LaunchConfig) (hK : 0<d.K) (hB : 0<c.B)
    (hone : c.one=oneTile d.N c.B) (row col : Nat) (hr : row<d.M)
    (hx : X∈regions) (s t : State α) (h : Agree regions (d.M*d.N*d.K) s t) :
    Agree regions (d.M*d.N*d.K) (FPGrid.forwardBlock A X Y d c row col s)
      (FPGrid.forwardBlock A X Y d c row col t) := by
  unfold FPGrid.forwardBlock
  split <;> cases hc : c.one
  · exact ForwardNoninner.graph_agree A regions X Y row col d.M d.N d.K c.B c.C hr hx s t h
  · exact scatter_agree regions _ s t h Y _ _ _ _
      (forward_noninner_value A regions X row col d.M d.N d.K c.B c.C hr hx s t h)
  · have hk : d.K=1 := by omega
    have hBN : c.B<d.N := by simp [hc,oneTile] at hone; omega
    simp only [hk,Nat.mul_one] at h ⊢
    exact ForwardInner.graph_agree A regions X Y row d.M d.N c.B hB hBN hr hx s t h
  · have hk : d.K=1 := by omega
    simp only [hk,Nat.mul_one] at h ⊢
    exact scatter_agree regions _ s t h Y _ _ _ _
      (forward_inner_value A regions X row d.M d.N c.B hr hx s t h)

theorem backward_block (A : Algebra α) (regions : List RegionName) (P G Y : RegionName)
    (d : Dimensions) (c : LaunchConfig) (hK : 0<d.K) (row col : Nat)
    (hr : 1<d.K → row<d.M) (hp : P∈regions) (hg : G∈regions)
    (s t : State α) (h : Agree regions (d.M*d.N*d.K) s t) :
    Agree regions (d.M*d.N*d.K) (FPGrid.backwardBlock A P G Y d c row col s)
      (FPGrid.backwardBlock A P G Y d c row col t) := by
  unfold FPGrid.backwardBlock
  split <;> cases hc : c.one
  · exact BackwardNoninner.graph_agree A regions P G Y row col d.M d.N d.K c.B c.C (hr ‹1<d.K›) hp hg s t h
  · exact scatter_agree regions _ s t h Y _ _ _ _
      (backward_noninner_value A regions P G row col d.M d.N d.K c.B c.C (hr ‹1<d.K›) hp hg s t h)
  · have hk : d.K=1 := by omega
    simp only [hk,Nat.mul_one] at h ⊢
    exact BackwardInner.graph_agree A regions P G Y row d.M d.N c.B c.C hp hg s t h
  · have hk : d.K=1 := by omega
    simp only [hk,Nat.mul_one] at h ⊢
    exact scatter_agree regions _ s t h Y _ _ _ _
      (backward_inner_value A regions P G row d.M d.N c.B c.C hp hg s t h)

theorem grid (regions : List RegionName) (n : Nat)
    (block : Nat → Nat → State α → State α) (ps : List Program)
    (step : ∀ p∈ps, ∀ s t,Agree regions n s t → Agree regions n (block p.1 p.2 s) (block p.1 p.2 t))
    (s t : State α) (h : Agree regions n s t) :
    Agree regions n (FPGrid.graphRun block ps s) (FPGrid.graphRun block ps t) := by
  induction ps generalizing s t with
  | nil => exact h
  | cons p ps ih =>
    exact ih (fun q hq => step q (by simp [hq])) _ _ (step p (by simp) s t h)

theorem forward (A : Algebra α) (regions : List RegionName) (X Y : RegionName)
    (d : Dimensions) (c : LaunchConfig) (hK : 0<d.K) (hB : 0<c.B)
    (hone : c.one=oneTile d.N c.B) (hx : X∈regions)
    (s t : State α) (h : Agree regions (d.M*d.N*d.K) s t) :
    Agree regions (d.M*d.N*d.K) (FPGrid.forwardGraph A X Y d c s) (FPGrid.forwardGraph A X Y d c t) := by
  apply grid
  · intro p member a b hab
    have hr : p.1<d.M := by
      unfold FPGrid.forwardGrid at member
      split at member <;> exact (mem_grid.mp member).1
    exact forward_block A regions X Y d c hK hB hone p.1 p.2 hr hx a b hab
  · exact h

theorem backward (A : Algebra α) (regions : List RegionName) (P G Y : RegionName)
    (d : Dimensions) (c : LaunchConfig) (hK : 0<d.K) (hp : P∈regions) (hg : G∈regions)
    (s t : State α) (h : Agree regions (d.M*d.N*d.K) s t) :
    Agree regions (d.M*d.N*d.K) (FPGrid.backwardGraph A P G Y d c s) (FPGrid.backwardGraph A P G Y d c t) := by
  apply grid
  · intro p member a b hab
    have hr : 1<d.K → p.1<d.M := by
      intro hk
      exact (mem_grid.mp (by simpa [FPGrid.backwardGrid,hk] using member)).1
    exact backward_block A regions P G Y d c hK p.1 p.2 hr hp hg a b hab
  · exact h

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality
