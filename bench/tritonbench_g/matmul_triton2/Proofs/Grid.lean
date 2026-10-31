import bench.tritonbench_g.matmul_triton2.Proofs.Logical
import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution
import bench.tritonbench_g.int_scaled_matmul.Proofs.Geometry
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def geometry (p : Parameters) : Geometry.Layout :=
  ⟨p.rows,p.columns,p.blockRows,p.blockColumns,p.groupRows,p.outputRow,p.outputColumn⟩
abbrev LogicalIndex (p : Parameters) := Geometry.Cell (geometry p)

theorem active_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : CIndex p) :
    active p (atProgram s pid) i ↔ Geometry.Active (geometry p) pid i := by rfl

theorem offset_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : CIndex p) :
    cOffset p (atProgram s pid) i=Geometry.offset (geometry p) pid i := by
  rfl

end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
