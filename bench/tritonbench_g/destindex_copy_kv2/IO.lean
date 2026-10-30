import bench.tritonbench_g.destindex_copy_kv2.Proofs.Tile
import bench.tritonbench_g.destindex_copy_kv1.IO

/-! The kv2 source shares kv1's finite typed row-copy IO: FP32 payloads, signed
destination indices, and indirect output rows. Only the actual invocation is
replaced here. The explicit width premise remains in the public theorem. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO
open VeriTile.Triton FP.Structural
noncomputable section

abbrev Inputs := DestindexCopyKv1.CopyIO.Inputs
abbrev Outputs := DestindexCopyKv1.CopyIO.Outputs
abbrev Admissible := @DestindexCopyKv1.CopyIO.Admissible
abbrev target := @DestindexCopyKv1.CopyIO.target

/-- Actual selected kv2 source and raw same-storage copy interpretation. -/
def invocation (rest : Algebra α) (c : Config) (pid : Nat) : Invocation α where
  kernel := kernel (layout c)
  backend := Tile.rawBackend rest
  pids := fun ax => if ax=0 then pid else 0

/-- Reuse the public finite source/metadata/output windows and their logical
scatter map; launch every program in the explicitly supplied schedule. -/
def copyIO (rest : Algebra α) (c : Config) (ps : List Nat) : ProgramIO α α where
  toMemoryIO := DestindexCopyKv1.CopyIO.memory c
  schedule := fun _ => ps.map (invocation rest c)
  encode := id
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO

namespace VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO
open VeriTile.Triton FP.Structural
noncomputable section

abbrev LogicalInputs := DestindexCopyKv1.CopyIO.LogicalInputs
abbrev LogicalOutputs := DestindexCopyKv1.CopyIO.LogicalOutputs
abbrev LogicalAdmissible := @DestindexCopyKv1.CopyIO.LogicalAdmissible

def logicalIO (rest : Algebra α) (c : Config) (ps : List Nat) :
    LogicalProgramIO α α (LogicalInputs α c) (LogicalOutputs α c) where
  core := copyIO rest c ps
  inputs := DestindexCopyKv1.CopyIO.inputEquiv α c
  outputs := DestindexCopyKv1.CopyIO.outputEquiv α c

end
end VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO
