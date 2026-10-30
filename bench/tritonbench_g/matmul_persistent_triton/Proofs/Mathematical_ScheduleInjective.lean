import bench.tritonbench_g.matmul_persistent_triton.Proofs.Mathematical_Schedule
import Mathlib.Data.Nat.ModEq

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.Schedule

/-- A source output row determines its group even when the last group has
fewer than GROUP_SIZE_M rows. -/
theorem group_of_row (rows columns group tile : Nat) (hg : 0 < group)
    (ht : tile < rows*columns) :
    row rows columns group tile / group = tile / (group*columns) := by
  have hb := grouped_bounds rows columns group tile hg ht
  have hr : tile % groupSize rows columns group tile < group :=
    lt_of_lt_of_le (Nat.mod_lt tile hb.1) (min_le_right _ _)
  unfold row groupBase
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hg, Nat.div_eq_of_lt hr, Nat.zero_add]

/-- The actual grouped source mapping never assigns the same row and column
to distinct output tiles. The absolute-tile modulo in pid_m is retained. -/
theorem grouped_injective (rows columns group tile tile' : Nat) (hg : 0 < group)
    (ht : tile < rows*columns) (ht' : tile' < rows*columns)
    (hr : row rows columns group tile = row rows columns group tile')
    (hc : column rows columns group tile = column rows columns group tile') : tile = tile' := by
  have hq : tile / (group*columns) = tile' / (group*columns) := by
    rw [← group_of_row rows columns group tile hg ht,
      ← group_of_row rows columns group tile' hg ht', hr]
  have hb : groupBase rows columns group tile = groupBase rows columns group tile' := by
    simp only [groupBase, hq]
  have hs : groupSize rows columns group tile = groupSize rows columns group tile' := by
    simp only [groupSize, hb]
  have hrem : tile % groupSize rows columns group tile = tile' % groupSize rows columns group tile := by
    unfold row at hr
    rw [← hb, ← hs] at hr
    exact Nat.add_left_cancel hr
  have hlocal : tile % (group*columns) % groupSize rows columns group tile =
      tile' % (group*columns) % groupSize rows columns group tile := by
    have h1 := Nat.div_add_mod tile (group*columns)
    have h2 := Nat.div_add_mod tile' (group*columns)
    rw [← hq] at h2
    have hglobal : (group*columns)*(tile/(group*columns)) + tile%(group*columns) ≡
        (group*columns)*(tile/(group*columns)) + tile'%(group*columns)
          [MOD groupSize rows columns group tile] := by
      simpa only [h1, h2, Nat.ModEq] using hrem
    exact Nat.ModEq.add_left_cancel' _ hglobal
  unfold column at hc
  rw [← hs] at hc
  exact Nat.ext_div_mod hq (Nat.ext_div_mod hc hlocal)

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.Schedule
