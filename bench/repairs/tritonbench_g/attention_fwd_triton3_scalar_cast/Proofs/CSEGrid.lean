import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.CSEProgram
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Grid
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Contract
import bench.tritonbench_g.context_attn_fwd.Proofs.HostFPComposition

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
namespace Composition
export VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostFPComposition
  (prepare schedule allPrograms allPrograms_complete)
end Composition
noncomputable section
variable {α : Type} [Inhabited α]

def launched (c : Call) : FP.ProfiledRewrite.Program := ⟨Kernels.keyMaskCSE c,[]⟩

/-- Execute every program using fresh registers and the actual preceding
program's memory. Successful execution and final-memory correspondence are
both conclusions, including empty grids and zero-iteration programs. -/
theorem complete_schedule (c : Call) (o : Ops α) (positive : 0<c.BN)
    (jobs : List (GridIndex c.grid)) (s : State α) :
    ∃ final,Composition.schedule o.algebra (launched c) c.grid jobs s=some final ∧
      final.mem=(referenceSchedule c o jobs s).mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons idx rest ih =>
    obtain ⟨mid,hmid,effect⟩ := exec_complete c o positive (Composition.prepare c.grid idx s)
    obtain ⟨final,hfinal,meaning⟩ := ih mid
    refine ⟨final,?_,?_⟩
    · change (exec (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) (Kernels.keyMaskCSE c) (Composition.prepare c.grid idx s)).bind _=some final
      rw [hmid]
      exact hfinal
    · rw [meaning]
      exact referenceSchedule_congr c o rest _ _ effect

theorem complete_grid (c : Call) (o : Ops α) (positive : 0<c.BN) (s : State α) :
    ∃ final,Composition.schedule o.algebra (launched c) c.grid
      (Composition.allPrograms c.grid) s=some final ∧ final.mem=(gridResult c o s).mem :=
  complete_schedule c o positive _ s

#axiomsClean complete_schedule
#axiomsClean complete_grid

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
