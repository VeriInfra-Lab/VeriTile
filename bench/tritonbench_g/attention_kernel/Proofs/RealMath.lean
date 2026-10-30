import VeriTile.Triton.Math.Attention
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.RealMath
open VeriTile.Triton
noncomputable section

def score (D BN : Nat) (scale : ℝ) (q k bias : Nat→Nat→ℝ) (m n : Nat) : ℝ :=
  (scale*1.44269504)*((Finset.range D).sum (fun d=>q m d*k n d))+
    (bias m (n/BN)+bias m (64+(n%BN)%64))*1.44269504

/-- Independent base-2 softmax weighted sum over the initial logical inputs.
The literal coefficient and both 128-wide bias windows are source-exact. -/
def expected (N D BN : Nat) (scale : ℝ) (q k v bias : Nat→Nat→ℝ) (m d : Nat) : ℝ :=
  ((Finset.range N).sum (fun n=>pow2 (score D BN scale q k bias m n)*v n d)) /
    ((Finset.range N).sum (fun n=>pow2 (score D BN scale q k bias m n)))

theorem denominator_positive (N D BN : Nat) (scale : ℝ) (q k bias : Nat→Nat→ℝ)
    (m : Nat) (hN : 0<N) :
    0<(Finset.range N).sum (fun n=>pow2 (score D BN scale q k bias m n)) := by
  exact Finset.sum_pos (fun _ _=>pow2_pos _) (by simpa using Nat.ne_of_gt hN)

theorem constant_value (N D BN : Nat) (scale value : ℝ) (q k bias : Nat→Nat→ℝ)
    (m d : Nat) (hN : 0<N) :
    expected N D BN scale q k (fun _ _=>value) bias m d=value := by
  unfold expected
  rw [←Finset.sum_mul]
  exact mul_div_cancel_left₀ value (ne_of_gt (denominator_positive N D BN scale q k bias m hN))

#specNonCircular expected avoiding [VeriTile.Triton.osBlockStep]
#axiomsClean denominator_positive
#axiomsClean constant_value
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.RealMath
