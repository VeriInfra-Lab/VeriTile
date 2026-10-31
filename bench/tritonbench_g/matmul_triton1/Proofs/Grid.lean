import bench.tritonbench_g.matmul_triton1.Proofs.Logical
import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution
import bench.tritonbench_g.int_scaled_matmul.Proofs.Geometry
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def geometry (p : Parameters) : Geometry.Layout :=
  ⟨p.rows,p.columns,p.blockRows,p.blockColumns,1,p.outputRow,p.outputColumn⟩
abbrev LogicalIndex (p : Parameters) := Geometry.Cell (geometry p)

theorem geometry_coordinates (p : Parameters) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) :
    Geometry.blockRow (geometry p) pid=pid/blocksN p ∧
    Geometry.blockColumn (geometry p) pid=pid%blocksN p := by
  have hpid : pid<blocksM p*blocksN p := hp
  have hn : 0<blocksN p := by
    by_contra h
    have hz : blocksN p=0 := by omega
    rw [hz,Nat.mul_zero] at hpid
    omega
  have hs := (GroupedGrid.facts (blocksM p) (blocksN p) 1 pid hn (by decide) hpid).2.1
  have he : GroupedGrid.size (blocksM p) (blocksN p) 1 pid=1 := by
    have hl := min_le_right (blocksM p-GroupedGrid.first 1 (blocksN p) pid) 1
    change GroupedGrid.size (blocksM p) (blocksN p) 1 pid≤1 at hl
    omega
  change GroupedGrid.row (blocksM p) (blocksN p) 1 pid=pid/blocksN p ∧
    GroupedGrid.col (blocksM p) (blocksN p) 1 pid=pid%blocksN p
  simp [GroupedGrid.row,GroupedGrid.col,GroupedGrid.first,he,Nat.mod_one]

theorem row_at {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) (i : CIndex p) :
    row p (atProgram s pid) i.1=Geometry.outRow (geometry p) pid i := by
  change pid/blocksN p*p.blockRows+i.1.val=Geometry.blockRow (geometry p) pid*p.blockRows+i.1.val
  rw [(geometry_coordinates p pid hp).1]

theorem column_at {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) (i : CIndex p) :
    column p (atProgram s pid) i.2.1=Geometry.outColumn (geometry p) pid i := by
  change pid%blocksN p*p.blockColumns+i.2.1.val=Geometry.blockColumn (geometry p) pid*p.blockColumns+i.2.1.val
  rw [(geometry_coordinates p pid hp).2]

theorem active_at {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) (i : CIndex p) :
    active p (atProgram s pid) i ↔ Geometry.Active (geometry p) pid i := by
  simp only [active,row_at p s pid hp i,column_at p s pid hp i]
  rfl

theorem offset_at {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (hp : pid<Geometry.count (geometry p)) (i : CIndex p) :
    cOffset p (atProgram s pid) i=Geometry.offset (geometry p) pid i := by
  simp only [cOffset,row_at p s pid hp i,column_at p s pid hp i]
  rfl

end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
