import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerWriteLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealLoopSafety

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton

 theorem range_aligned_safe
    {bounds : RegionBounds} {idx : RegName} {q B : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} (hB : 0 < B)
    (hsafe : ∀ c s, c < q → P c s →
      Stmt.TraceSafeList bounds body (s.setReg idx .nat [] (Tile.scalar (c * B))))
    (hstep : ∀ c s, c < q → P c s →
      ∃ s', stepStmts body (s.setReg idx .nat [] (Tile.scalar (c * B))) = some s' ∧
        P (c + 1) s') :
    ∀ c s, c ≤ q → P c s →
      Stmt.forRangeTraceSafe bounds idx (c * B) (q * B) B body s := by
  intro c s
  induction h : (q - c) using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hc hp
    by_cases hdone : c = q
    · subst c
      rw [Stmt.forRangeTraceSafe, if_neg (Nat.ne_of_gt hB), if_neg (Nat.lt_irrefl _)]
      trivial
    · have hlt : c < q := by omega
      obtain ⟨next, hbody, hnext⟩ := hstep c s hlt hp
      rw [Stmt.forRangeTraceSafe, if_neg (Nat.ne_of_gt hB),
        if_pos (Nat.mul_lt_mul_of_pos_right hlt hB)]
      refine ⟨hsafe c s hlt hp, ?_⟩
      rw [hbody]
      have ht := ih (q - (c + 1)) (by omega) (c + 1) next rfl (by omega) hnext
      simpa only [Nat.add_mul, Nat.one_mul] using ht

 theorem row_address_bound {row M N n : Nat} (hr : row < M) (hn : n < N) :
    row * N + n < M * N := by
  simpa using RealAddress.non_inner_bound hr (show 0 < 1 by omega) hn

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

#guard_msgs (drop info) in
#auditModuleAxioms
