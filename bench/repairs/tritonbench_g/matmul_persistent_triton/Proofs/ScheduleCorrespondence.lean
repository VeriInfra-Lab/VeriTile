import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.ControlBounds
import bench.tritonbench_g.matmul_persistent_triton.Proofs.IterationSchedule
import bench.tritonbench_g.matmul_persistent_triton.Proofs.FlatSchedule

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
set_option maxHeartbeats 5000000

/-- Bounding the group width avoids an oversized intermediate while leaving
every actual grouped tile coordinate in the original order. -/
theorem original_group_order (p : Problem) (tile : Nat) (ht : tile<p.tiles) :
    p.tileRow tile=Schedule.row p.rows p.columns 8 tile ∧
    p.tileColumn tile=Schedule.column p.rows p.columns 8 tile := by
  by_cases large : 8≤p.rows
  · simp only [Problem.tileRow,Problem.tileColumn,Problem.group,min_eq_left large,and_self]
  · have small : p.rows≤8 := by omega
    have bound : tile<p.rows*p.columns := ht
    have bound8 : tile<8*p.columns := lt_of_lt_of_le bound (Nat.mul_le_mul_right p.columns small)
    simp only [Problem.tileRow,Problem.tileColumn,Problem.group,min_eq_right small,
      Schedule.row,Schedule.column,Schedule.groupBase,Schedule.groupSize,
      Nat.div_eq_of_lt bound,Nat.div_eq_of_lt bound8,Nat.zero_mul,Nat.sub_zero,
      min_self,min_eq_left small,Nat.mod_eq_of_lt bound,Nat.mod_eq_of_lt bound8,Nat.zero_add,and_self]

/-- The repaired nested loops enumerate exactly the old flat loop's tile/K
coordinates. This flat index is proof-only; the candidate never evaluates
the potentially oversized work*K product on device. -/
theorem nested_flat_coordinates (p : Problem) (pid ordinal iteration : Nat)
    (hk : 0<p.K) (hq : ordinal<p.work pid) (hi : iteration<p.kTiles) :
    (ordinal*p.kTiles+iteration)/p.kTiles=ordinal ∧
    (ordinal*p.kTiles+iteration)%p.kTiles=iteration ∧
    ordinal*p.kTiles+iteration<p.kTiles*p.work pid := by
  have positive : 0<p.kTiles := RealMath.blocks_positive p.K p.profile.blockK hk (blockK_positive p.profile)
  have mul := Nat.mul_le_mul_right p.kTiles (show ordinal+1≤p.work pid by omega)
  refine ⟨?_,?_,?_⟩
  · rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hi,Nat.zero_add]
  · simp [Nat.add_mod,Nat.mod_eq_of_lt hi]
  · nlinarith

theorem nested_inner_next (k ordinal iteration : Nat) :
    ordinal*k+(iteration+1)=(ordinal*k+iteration)+1 := by omega

theorem nested_tile_next (k ordinal : Nat) (hk : 0<k) :
    (ordinal+1)*k=(ordinal*k+(k-1))+1 := by
  rw [Nat.add_mul,Nat.one_mul]
  omega

#axiomsClean original_group_order
#axiomsClean nested_flat_coordinates
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatmulPersistentTriton
