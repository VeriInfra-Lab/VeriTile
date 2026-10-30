import bench.repairs.tritonbench_g.chunk_gla_fwd.IO
import VeriTile.Triton.Float.FillProgramIO
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.KernelProfile
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPReadBounds

/-! Concrete Structural invocations for the existing native launch order.
No storage-view conversion is part of this executor: successors retain the
actual typed cells written by their predecessors. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeLaunch
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open FPStages (Stage)
noncomputable section
variable {α : Type} [Inhabited α]



theorem initial_eq (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (s : State α) : (job M D stage idx).initial s=start stage idx s := rfl






/-- Exact existing native order, including the empty-producer skip and the
large-K scratch initialization after the inter stage. -/



theorem program_run (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (s : State α) :
    ∃ t,(job M D stage idx).run s=some t ∧
      t.mem=(FPStages.program M D stage (start stage idx s)).mem :=
  FPStages.program_correct M D stage (start stage idx s)

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeLaunch
