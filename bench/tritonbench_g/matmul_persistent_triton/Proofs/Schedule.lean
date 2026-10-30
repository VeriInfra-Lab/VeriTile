import bench.tritonbench_g.matmul_persistent_triton.Proofs.Parameters
import Mathlib.Tactic

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.Schedule

/-- The quotient/remainder work distribution from the original kernel. -/
def tilesFor (total sms pid : Nat) : Nat :=
  total / sms + if pid < total % sms then 1 else 0

def tileId (sms pid ordinal : Nat) : Nat := pid + ordinal * sms

def groupBase (rows columns group tile : Nat) : Nat :=
  tile / (group * columns) * group

def groupSize (rows columns group tile : Nat) : Nat :=
  min (rows - groupBase rows columns group tile) group

def row (rows columns group tile : Nat) : Nat :=
  groupBase rows columns group tile + tile % groupSize rows columns group tile

def column (rows columns group tile : Nat) : Nat :=
  (tile % (group * columns)) / groupSize rows columns group tile

/-- Every iteration assigned to one launch PID addresses a legal output tile,
and every tile of its residue class occurs before that PID's loop ends. -/
theorem ordinal_bound_iff (total sms pid q : Nat) (hs : 0 < sms) (hp : pid < sms) :
    q < tilesFor total sms pid ↔ tileId sms pid q < total := by
  have hm := Nat.mod_lt total hs
  have hdecomp : total % sms + (total / sms) * sms = total := by
    simpa only [Nat.mul_comm] using Nat.mod_add_div total sms
  by_cases hq : q < total / sms
  · have hmul := Nat.mul_le_mul_right sms (Nat.succ_le_of_lt hq)
    simp only [Nat.succ_mul, Nat.add_mul, Nat.one_mul] at hmul
    unfold tilesFor tileId
    split_ifs <;> constructor <;> intro _ <;> omega
  · by_cases heq : q = total / sms
    · subst q
      unfold tilesFor tileId
      split_ifs <;> omega
    · have hmul := Nat.mul_le_mul_right sms
        (show total / sms + 1 ≤ q by omega)
      simp only [Nat.succ_mul, Nat.add_mul, Nat.one_mul] at hmul
      unfold tilesFor tileId
      split_ifs <;> omega

/-- The host grid min(SMS,total) covers every tile exactly once by quotient
and remainder. A zero-tile grid needs no fictitious active PID. -/
theorem tile_covered (total sms tile : Nat) (hs : 0 < sms) (ht : tile < total) :
    tile % sms < min sms total ∧
    tile / sms < tilesFor total sms (tile % sms) ∧
    tileId sms (tile % sms) (tile / sms) = tile := by
  have hm := Nat.mod_lt tile hs
  have hle := Nat.mod_le tile sms
  have heq : tileId sms (tile % sms) (tile / sms) = tile := by
    simpa only [tileId, Nat.mul_comm] using Nat.mod_add_div tile sms
  refine ⟨lt_min hm (by omega), ?_, heq⟩
  exact (ordinal_bound_iff total sms (tile % sms) (tile / sms) hs hm).mpr (by rw [heq]; exact ht)

theorem tile_unique (sms p q p' q' : Nat) (hs : 0 < sms)
    (hp : p < sms) (hp' : p' < sms)
    (heq : tileId sms p q = tileId sms p' q') : p = p' ∧ q = q' := by
  have hrem := congrArg (fun n => n % sms) heq
  simp [tileId, Nat.add_mod, Nat.mod_eq_of_lt hp, Nat.mod_eq_of_lt hp'] at hrem
  subst p'
  simp only [tileId] at heq
  exact ⟨rfl, Nat.eq_of_mul_eq_mul_right hs (Nat.add_left_cancel heq)⟩

/-- Even a shortened final group has a positive active row count. -/
theorem grouped_bounds (rows columns group tile : Nat)
    (hg : 0 < group) (ht : tile < rows * columns) :
    0 < groupSize rows columns group tile ∧
    row rows columns group tile < rows ∧
    column rows columns group tile < columns := by
  have hc : 0 < columns := by nlinarith
  have hmul := Nat.div_mul_le_self tile (group * columns)
  have hb : groupBase rows columns group tile < rows := by
    unfold groupBase
    nlinarith
  have hsize : 0 < groupSize rows columns group tile := by
    unfold groupSize
    exact lt_min (by omega) hg
  have hspan : groupBase rows columns group tile +
      groupSize rows columns group tile ≤ rows := by
    have hh := min_le_left (rows - groupBase rows columns group tile) group
    unfold groupSize
    omega
  have hrem : tile % (group * columns) <
      groupSize rows columns group tile * columns := by
    unfold groupSize
    by_cases hsmall : rows - groupBase rows columns group tile ≤ group
    · rw [min_eq_left hsmall]
      have hd := Nat.mod_add_div tile (group * columns)
      have hs := Nat.sub_add_cancel (Nat.le_of_lt hb)
      unfold groupBase at hs ⊢
      nlinarith
    · rw [min_eq_right (by omega)]
      exact Nat.mod_lt tile (Nat.mul_pos hg hc)
  refine ⟨hsize, ?_, ?_⟩
  · have hr := Nat.mod_lt tile hsize
    unfold row
    omega
  · unfold column
    exact (Nat.div_lt_iff_lt_mul hsize).mpr (by simpa [Nat.mul_comm] using hrem)

/-- The signed negative initialization reaches the first tile without natural
subtraction truncation, including PIDs strictly below the SMS count. -/
theorem initial_tile (sms pid : Nat) :
    ((pid : Int) - sms) + sms = (tileId sms pid 0 : Int) := by
  simp [tileId]

theorem first_ki (kTiles : Nat) (hk : 0 < kTiles) :
    (if (-1 : Int) = (kTiles : Int) - 1 then 0 else (-1 : Int) + 1) = 0 := by
  split_ifs <;> omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.Schedule
