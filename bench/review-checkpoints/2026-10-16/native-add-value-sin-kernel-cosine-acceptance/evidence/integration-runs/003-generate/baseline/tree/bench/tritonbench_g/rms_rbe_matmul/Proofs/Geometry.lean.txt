import bench.tritonbench_g.rms_rbe_matmul.Proofs.LegacyReal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.Geometry
open VeriTile.Triton

/-- Padded row-major output is injective on active lanes. The batch offset
is fixed per program and N may be smaller than the block width BN. -/
theorem active_injective (s : BlockState) (M N BM BN sob som son : Nat)
    (hcol : 0 < son) (hrow : N * son ≤ som)
    (i j : TileIndex [BM, BN]) (hi : active s M N BM BN i) (hj : active s M N BM BN j)
    (he : outOffset s N BM BN sob som son i = outOffset s N BM BN sob som son j) : i = j := by
  have hci : son * colGlobal s N BN i.2.1 < som :=
    lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_left hi.2 hcol) (by simpa [Nat.mul_comm] using hrow)
  have hcj : son * colGlobal s N BN j.2.1 < som :=
    lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_left hj.2 hcol) (by simpa [Nat.mul_comm] using hrow)
  have he' : som * rowGlobal s N BM BN i.1 + son * colGlobal s N BN i.2.1 =
      som * rowGlobal s N BM BN j.1 + son * colGlobal s N BN j.2.1 := by
    simpa [outOffset, Nat.add_assoc, Nat.mul_comm] using Nat.add_left_cancel
      (show s.pids 0 * sob + (rowGlobal s N BM BN i.1 * som + colGlobal s N BN i.2.1 * son) =
        s.pids 0 * sob + (rowGlobal s N BM BN j.1 * som + colGlobal s N BN j.2.1 * son) by
        simpa [outOffset, Nat.add_assoc] using he)
  have hr : rowGlobal s N BM BN i.1 = rowGlobal s N BM BN j.1 := by
    by_contra hne
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · have hb := Nat.mul_le_mul_left som hlt
      simp only [Nat.mul_succ] at hb
      omega
    · have hb := Nat.mul_le_mul_left som hgt
      simp only [Nat.mul_succ] at hb
      omega
  rw [hr] at he'
  have hc : colGlobal s N BN i.2.1 = colGlobal s N BN j.2.1 :=
    Nat.eq_of_mul_eq_mul_left hcol (Nat.add_left_cancel he')
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel hr)
  · apply Prod.ext
    · exact Fin.ext (Nat.add_left_cancel hc)
    · exact Subsingleton.elim _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.Geometry
