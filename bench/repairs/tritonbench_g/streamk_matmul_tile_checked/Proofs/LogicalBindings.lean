import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.IO
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.KernelResults

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.LogicalIO
open Whole001 Source Checked
noncomputable section

def snapshot (p : Parameters) (s : State α p exterior) : Inputs α p :=
  ⟨s.inputA,s.inputB,s.output,s.scratch⟩

def initial (p : Parameters) (values : Inputs α p) (outside : Fin exterior → α) : State α p exterior :=
  ⟨values.a,values.b,values.initialOutput,values.initialScratch,outside⟩

theorem snapshot_initial (p : Parameters) (values : Inputs α p) (outside : Fin exterior → α) :
    snapshot p (initial p values outside)=values := rfl

theorem initial_snapshot (p : Parameters) (s : State α p exterior) :
    initial p (snapshot p s) s.exterior=s := rfl

theorem real_snapshot (p : Parameters) (s : State ℝ p exterior) (cell : Returned.Cell p) :
    realFormula p (snapshot p s) cell=Reference.matrixProduct p s.inputA s.inputB cell.1 cell.2 := rfl

theorem native_snapshot (reference : Native.Reference) (p : Parameters)
    (s : State Native.Word p exterior) (cell : Returned.Cell p) :
    nativeFormula reference p (snapshot p s) cell=
      Returned.value p (Native.operations (Native.referenceInstructions reference)) s.inputA s.inputB cell := rfl

#axiomsClean initial_snapshot
#axiomsClean real_snapshot
#axiomsClean native_snapshot
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.LogicalIO
