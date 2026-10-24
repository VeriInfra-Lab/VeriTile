import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Real
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.ControlBounds
import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Source

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 5000000

/-- The original kernel uses the local PID remainder before its row remainder.
Shortening GROUP8 when fewer than eight tile rows exist preserves both coordinates. -/
theorem original_group_order (p : Problem) (tile : Nat) (ht : tile<p.tiles) :
    p.tileRow tile=Schedule.groupBase p.rows p.columns 8 tile+
      (tile%(8*p.columns))%Schedule.groupSize p.rows p.columns 8 tile ∧
    p.tileColumn tile=Schedule.column p.rows p.columns 8 tile := by
  by_cases large : 8≤p.rows
  · have group : p.group=8 := Nat.min_eq_left large
    simp [Problem.tileRow,Problem.tileColumn,Problem.groupBase,Problem.groupSize,group]
  · have group : p.group=p.rows := Nat.min_eq_right (by omega)
    have small : p.rows≤8 := by omega
    have oldBound : tile<8*p.columns := lt_of_lt_of_le ht (Nat.mul_le_mul_right p.columns small)
    have newBound : tile<p.rows*p.columns := ht
    simp [Problem.tileRow,Problem.tileColumn,Problem.groupBase,Problem.groupSize,group,
      Schedule.groupBase,Schedule.groupSize,Schedule.column,Nat.div_eq_of_lt oldBound,
      Nat.mod_eq_of_lt oldBound,Nat.div_eq_of_lt newBound,Nat.mod_eq_of_lt newBound,
      Nat.min_eq_left small]

/-- The repair Real target is exactly the original independent logical target
when logical input coordinates refer to the admitted raw/resolved tensor cells. -/
theorem original_mathematical (p : Problem) (b : Buffers) (entry : Memory ℝ)
    (row : Fin p.M) (column : Fin p.N) :
    Real.formula p b entry row.val column.val=
      VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.SourceScope.mathematical
        p.M p.N p.K p.enabled
        (fun i k => entry (b.A.address i.val k.val))
        (fun k j => entry (b.B.address k.val j.val)) row column := by
  unfold Real.formula Real.activate RealMath.formula Real.inputs
  unfold VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.SourceScope.mathematical
  dsimp only
  rw [Fin.sum_univ_eq_sum_range (fun k => entry (b.A.address row.val k)*entry (b.B.address k column.val))]

/-- The sole configuration retains all semantic tile constants and warps.
Its pipeline stage count is explicitly repaired from three to two because
the recorded FP32 stages3 artifact exceeds SM90 shared-memory resources. -/
theorem resource_configuration :
    TypedSource.configuration=(128,256,64,8,2,8) ∧
    Storage.stages .fp32=2 ∧
    VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.SourceScope.stages=3 := by
  decide

#axiomsClean original_group_order
#axiomsClean original_mathematical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8
