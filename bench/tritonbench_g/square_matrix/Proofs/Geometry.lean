import bench.tritonbench_g.square_matrix.Contract
import VeriTile.Meta.StatementAudit
import Mathlib

namespace VeriTile.Bench.Repairs.SquareMatrix
theorem block_covers (n : Nat) : n ≤ block n := by
  by_cases hn : n=0
  · simp [hn,block]
  · simp only [block,if_neg hn]
    have h1 := Nat.left_le_or (n:=n-1) (m:=(n-1) >>> 1)
    have h2 := Nat.left_le_or (n:=(n-1) ||| ((n-1) >>> 1))
      (m:=((n-1) ||| ((n-1) >>> 1)) >>> 2)
    generalize (n-1) ||| ((n-1) >>> 1) ||| (((n-1) ||| ((n-1) >>> 1)) >>> 2) = a at *
    have h3 := Nat.left_le_or (n:=a) (m:=a >>> 4)
    generalize a ||| (a >>> 4) = b at *
    have h4 := Nat.left_le_or (n:=b) (m:=b >>> 8)
    generalize b ||| (b >>> 8) = c at *
    have h5 := Nat.left_le_or (n:=c) (m:=c >>> 16)
    generalize c ||| (c >>> 16) = d at *
    have h6 := Nat.left_le_or (n:=d) (m:=d >>> 32)
    omega

theorem warps_low (B : Nat) (h : B<2048) : warps B=4 := by
  simp [warps,show ¬2048≤B by omega,show ¬4096≤B by omega]
theorem warps_middle (B : Nat) (h1 : 2048≤B) (h2 : B<4096) : warps B=8 := by simp [warps,h1]; omega
theorem warps_high (B : Nat) (h : 4096≤B) : warps B=16 := by simp [warps,h]


theorem flat_lt (m n r c : Nat) (hr : r < m) (hc : c < n) : r*n+c < m*n := by
  have h := Nat.mul_le_mul_right n (Nat.succ_le_of_lt hr)
  rw [Nat.succ_mul] at h
  omega

theorem row_unique (n r c k : Nat) (hc : c < n)
    (lo : k*n ≤ r*n+c) (hi : r*n+c < k*n+n) : k=r := by
  have pos : 0 < n := by omega
  nlinarith

#axiomsClean block_covers
#axiomsClean flat_lt
#axiomsClean row_unique
end VeriTile.Bench.Repairs.SquareMatrix
