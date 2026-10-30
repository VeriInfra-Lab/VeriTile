import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.kcache_copy_triton.IO
import bench.tritonbench_g.kcache_copy_triton.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.LogicalReader
open VeriTile.Triton FP.Structural
open scoped ProgramIO
noncomputable section

/-- The actual source copies one full (token, head, split) tile. Length and
selected-block metadata are signed integer input channels; their dependent
windows retain arbitrary admitted aliases. The FP32 payload channel has its
mathematical Real interpretation here. Every load/store is active. -/
abbrev copyIO (p : Source.Parameters) (ids : Nat → Nat) : ProgramIO ℝ ℝ :=
  CopyIO.io CopyIO.realBackend Bool.false p ids

theorem kcache_copy_triton_real (p : Source.Parameters) (ids : Nat → Nat) :
    Spec.Real (copyIO p ids ⊨{CopyIO.OutputInjective p ids}
      fun vals o => Fin.cases (fun j => vals 2 j) (fun z => Fin.elim0 z) o) :=
  CopyIO.implements CopyIO.realBackend Bool.false p ids CopyIO.formula (fun _ => rfl)

#stmtSurfaceSubset kcache_copy_triton_real ⊆
  [Source.Parameters,Spec.Real,copyIO,ProgramIO.ImplementsOn,CopyIO.OutputInjective,
   ProgramIO.Inputs, IOChannel, MemoryIO.outputs, ProgramIO.toMemoryIO, Values, IOChannel.dtype,
   ChannelValues._proof_2, IOChannel.shape, TileIndex, MemoryIO.inputs, List.instNeZeroNatLengthCons,
   IOChannel.mk, TileDType.int, Region.cast, TileDType.real, Source.Parameters.lengths,
   Source.Parameters.blockTables, Source.Parameters.width, Source.Parameters.K]
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.LogicalReader

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.LogicalReader
open VeriTile.Triton FP.Structural
open scoped LogicalProgramIO
noncomputable section

theorem full_judgment_iff (p : Source.Parameters) (ids : Nat → Nat) :
    (CopyIO.logicalIO CopyIO.realBackend Bool.false p ids ⊨{CopyIO.logicalOutputInjective p ids}
      fun xs lane => xs.payload lane) ↔ (copyIO p ids).ImplementsOn (CopyIO.OutputInjective p ids)
        (fun (vals : (copyIO p ids).Inputs) o => Fin.cases (fun j => vals 2 j) (fun z => Fin.elim0 z) o) := by
  have target : (CopyIO.logicalIO CopyIO.realBackend Bool.false p ids).lift (fun xs lane => xs.payload lane) =
      (fun (vals : (copyIO p ids).Inputs) o => Fin.cases (fun j => vals 2 j) (fun z => Fin.elim0 z) o) := by
    funext xs o; fin_cases o; funext j; rfl
  change (copyIO p ids).ImplementsOn (CopyIO.OutputInjective p ids) _ ↔ _
  rw [target]
  rfl

end
end VeriTile.Bench.Repairs.KcacheCopyTriton.LogicalReader
