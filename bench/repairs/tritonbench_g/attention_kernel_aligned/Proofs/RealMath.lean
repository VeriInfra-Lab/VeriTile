import VeriTile.Triton.Math.Attention
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.RealMath
open VeriTile.Triton
noncomputable section
/-- Independent noncausal base-2 bias attention over exactly N keys. -/
def score (D BN biasLast : Nat) (scale : ℝ) (q k bias : Nat → Nat → ℝ) (m j : Nat) : ℝ :=
  (∑ d∈Finset.range D,(q m d*(scale*1.44269504))*k j d) +
    (bias m (j/BN)+bias m (j%BN%biasLast+biasLast))
def expected (N D BN biasLast : Nat) (scale : ℝ)
    (q k v bias : Nat → Nat → ℝ) (m d : Nat) : ℝ :=
  (∑ j∈Finset.range N,pow2 (score D BN biasLast scale q k bias m j)*v j d) /
    (∑ j∈Finset.range N,pow2 (score D BN biasLast scale q k bias m j))
theorem score_scaled_dot (D BN biasLast : Nat) (scale : ℝ)
    (q k bias : Nat → Nat → ℝ) (m j : Nat) :
    score D BN biasLast scale q k bias m j=
      (scale*1.44269504)*(∑ d∈Finset.range D,q m d*k j d)+
        (bias m (j/BN)+bias m (j%BN%biasLast+biasLast)) := by
  unfold score
  rw [Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro d hd
  ring

def prefixSum (N t : Nat) (f : Nat → ℝ) : ℝ := ∑ j∈Finset.range t,if j<N then f j else 0
def chunk (BN N row : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ j : Fin BN,if row+j.val<N then f (row+j.val) else 0
theorem prefix_step (BN N row : Nat) (f : Nat → ℝ) :
    prefixSum N (row+BN) f=prefixSum N row f+chunk BN N row f := by
  simp only [prefixSum,chunk,Finset.sum_range_add]
  rw [Fin.sum_univ_eq_sum_range (fun j=>if row+j<N then f (row+j) else 0) BN]
theorem prefix_done (N row : Nat) (f : Nat → ℝ) (h : N≤row) :
    prefixSum N row f=∑ j∈Finset.range N,f j := by
  rw [show row=N+(row-N) by omega,prefixSum,Finset.sum_range_add]
  have first : (∑ j∈Finset.range N,if j<N then f j else 0)=∑ j∈Finset.range N,f j := by
    apply Finset.sum_congr rfl
    intro j hj
    simp [Finset.mem_range.mp hj]
  simpa [first]
#axiomsClean score_scaled_dot
#axiomsClean prefix_done
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.RealMath
