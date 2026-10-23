import bench.repairs.tritonbench_g.int4_matmul.PairIO
import bench.repairs.tritonbench_g.int4_matmul.Proofs.FoldStride

namespace VeriTile.Bench.Repairs.Int4Matmul.FoldStrideIO
open VeriTile Triton FP.Structural KernelExecution
noncomputable section
variable {α : Type} [Inhabited α]

theorem invocation_run (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (p : IO.Program) (s : FP.Structural.State (SourceGraph.Node α)) :
    (IO.invocation x t o rest p).run s=(invocation x t o rest p).run s :=
  FoldStride.execution (SourceGraph.engine o rest) (Raw.parameters x t) ((IO.invocation x t o rest p).initial s)

theorem invocation_safe (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (p : IO.Program) (bounds : RegionName → Nat) (s : FP.Structural.State (SourceGraph.Node α)) :
    (IO.invocation x t o rest p).Safe bounds s ↔ (invocation x t o rest p).Safe bounds s :=
  FoldStride.safety (SourceGraph.engine o rest) (Raw.parameters x t) bounds ((IO.invocation x t o rest p).initial s)

theorem equivalent (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (complete : lanes.Perm (rawLanes x.shape t)) :
    (IO.programIO x t o rest lanes).EquivalentOnCells (variant x t o rest lanes) (IO.domain x.shape o) :=
  .of_run_safe (variant x t o rest lanes) rfl
    (fun _ p s=>invocation_run x t o rest p s)
    (fun _ p bounds s=>invocation_safe x t o rest p bounds s)
    (IOProof.implements x geometry t member o rest lanes complete)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.Int4Matmul.FoldStrideIO
