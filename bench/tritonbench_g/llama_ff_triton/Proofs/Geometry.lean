import bench.tritonbench_g.llama_ff_triton.Proofs.LegacyReal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.Geometry
open VeriTile.Triton

/-- Contiguous or padded row-major output is injective on active lanes even
when N < BN. No claim is made about addresses of masked-out lanes. -/
theorem active_injective (s : BlockState) (M N BM BN soutm soutn : Nat)
    (hcol : 0 < soutn) (hrow : N * soutn ≤ soutm)
    (i j : TileIndex [BM, BN]) (hi : active s M N BM BN i) (hj : active s M N BM BN j)
    (he : outOffset s N BM BN soutm soutn i = outOffset s N BM BN soutm soutn j) : i = j := by
  have hci : soutn * colGlobal s N BN i.2.1 < soutm :=
    lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_left hi.2 hcol) (by simpa [Nat.mul_comm] using hrow)
  have hcj : soutn * colGlobal s N BN j.2.1 < soutm :=
    lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_left hj.2 hcol) (by simpa [Nat.mul_comm] using hrow)
  change soutm * rowGlobal s N BM BN i.1 + soutn * colGlobal s N BN i.2.1 =
    soutm * rowGlobal s N BM BN j.1 + soutn * colGlobal s N BN j.2.1 at he
  have hr : rowGlobal s N BM BN i.1 = rowGlobal s N BM BN j.1 := by
    by_contra hne
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · have hb := Nat.mul_le_mul_left soutm hlt
      simp only [Nat.mul_succ] at hb
      omega
    · have hb := Nat.mul_le_mul_left soutm hgt
      simp only [Nat.mul_succ] at hb
      omega
  rw [hr] at he
  have hc : colGlobal s N BN i.2.1 = colGlobal s N BN j.2.1 :=
    Nat.eq_of_mul_eq_mul_left hcol (Nat.add_left_cancel he)
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel hr)
  · apply Prod.ext
    · exact Fin.ext (Nat.add_left_cancel hc)
    · exact Subsingleton.elim _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.Geometry
