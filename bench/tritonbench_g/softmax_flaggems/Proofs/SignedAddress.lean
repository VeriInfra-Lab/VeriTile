import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealAddress
import Mathlib.Data.Int.Init

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.SignedAddress
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress

/-- The signed helper result agrees with the nonnegative tile origin precisely
on the wrapper's nonempty launch domain. In particular N=0 is not clipped here. -/
theorem previous_eq (N B : Nat) (hN : 0<N) (hB : 0<B) :
    (((N : Int)+(B : Int)-1)/(B : Int))*(B : Int)-(B : Int) =
      (RealAddress.previous N B : Int) := by
  have numerator : 1≤N+B := by omega
  have product : B≤rounds N B*B := Nat.le_mul_of_pos_left B (rounds_positive hN hB)
  have hn : ((N+B-1 : Nat) : Int) = (N : Int)+(B : Int)-1 := by omega
  rw [←hn,←Int.natCast_ediv,←Int.natCast_mul]
  exact (Int.ofNat_sub product).symm

/-- Every actually visited tile is at or before the reverse-write origin. -/
theorem tile_le_previous (N B cur : Nat) (hN : 0<N) (hB : 0<B)
    (hcur : cur<N) (aligned : B∣cur) : cur≤RealAddress.previous N B := by
  obtain ⟨q,rfl⟩ := aligned
  have hq : q<rounds N B := (tile_start_lt_iff hN hB).mp (by simpa [Nat.mul_comm] using hcur)
  rw [RealAddress.previous_eq]
  have : q≤rounds N B-1 := by omega
  simpa [Nat.mul_comm] using Nat.mul_le_mul_right B this

/-- Signed subtraction precedes the address cast, including the lane offset. -/
theorem reverse_lane (previous cur lane : Nat) (h : cur≤previous) :
    (((previous : Int)-(cur : Int))+(lane : Int)).toNat = previous-cur+lane := by
  rw [←Int.ofNat_sub h,←Int.natCast_add,Int.toNat_natCast]
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.SignedAddress
