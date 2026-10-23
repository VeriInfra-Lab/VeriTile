import bench.repairs.tritonbench_g.destindex_copy.Kernels
import bench.repairs.tritonbench_g.destindex_copy_kv1.IO

namespace VeriTile.Bench.Repairs.DestindexCopy
open VeriTile.Triton FP.Structural
open DestindexCopyKv1
noncomputable section

/-- Reuse the exact shared copy descriptor: FP32 payload[count,heads,dims],
signed destination[count], finite scatter outputs and a full source schedule.
All fields are declared once in destindex_copy_kv1/IO.lean. -/
abbrev copyIO (rest : Algebra α) (c : Config) (ps : List Nat) : ProgramIO α α :=
  CopyIO.copyIO rest c ps

/-- Each reused invocation runs this task's selected kernel, for either plane. -/
theorem source_identity (rest : Algebra α) (c : Config) (pid : Nat) :
    (CopyIO.invocation rest c pid).kernel=kernel (layout c) := rfl

end
end VeriTile.Bench.Repairs.DestindexCopy

namespace VeriTile.Bench.Repairs.DestindexCopy
open VeriTile.Triton FP.Structural
open DestindexCopyKv1
noncomputable section

/-- Either plane uses the exact shared complete copy grid and signed metadata. -/
abbrev logicalCopyIO (rest : Algebra ℝ) (c : Config) (ps : List Nat) :=
  CopyIO.logicalIO rest c ps

end
end VeriTile.Bench.Repairs.DestindexCopy
