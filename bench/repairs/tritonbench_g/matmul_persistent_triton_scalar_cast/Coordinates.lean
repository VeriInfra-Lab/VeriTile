import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Model

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Signed-coordinate calculations are exact on each legal persistent tile. -/
theorem coordinates (p : Problem) (tile : Nat) (ht : tile<p.tiles) :
    let g := (tile : Int)/((p.group*p.columns : Nat) : Int)
    let first := g*(p.group : Int)
    let size := min ((p.rows : Int)-first) (p.group : Int)
    (first+(tile : Int)%size : Int)=(p.tileRow tile : Nat) ∧
      ((tile : Int)%((p.group*p.columns : Nat) : Int)/size : Int)=(p.tileColumn tile : Nat) := by
  dsimp only
  have hm : 0<p.M := by
    by_contra hn
    have hz : p.M=0 := by omega
    simp [Problem.tiles,Problem.rows,VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealMath.blocks,hz] at ht
  have bounds := Schedule.grouped_bounds p.rows p.columns p.group tile (p.group_positive hm) ht
  have base : Schedule.groupBase p.rows p.columns p.group tile≤p.rows := by
    have hmin := Nat.min_le_left (p.rows-Schedule.groupBase p.rows p.columns p.group tile) p.group
    unfold Schedule.groupSize at bounds
    omega
  have castSub : (p.rows : Int)-(Schedule.groupBase p.rows p.columns p.group tile : Nat)=
      ((p.rows-Schedule.groupBase p.rows p.columns p.group tile : Nat) : Int) := by omega
  simp only [Problem.tileRow,Problem.tileColumn,Schedule.row,Schedule.column,Schedule.groupSize]
  simp only [Schedule.groupBase] at base castSub ⊢
  norm_cast

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
