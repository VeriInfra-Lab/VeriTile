import bench.tritonbench_g.attention_fwd_triton3.Proofs.Program
import bench.tritonbench_g.attention_fwd_triton3.Contract
import bench.tritonbench_g.context_attn_fwd.Proofs.Mathematical_HostFPComposition

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
namespace Composition
export VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostFPComposition
  (prepare schedule allPrograms allPrograms_complete)
end Composition
noncomputable section
variable {α : Type} [Inhabited α]

def launched (c : Call) : FP.ProfiledRewrite.Program := ⟨Kernels.source c,[]⟩

/-- Independent grid target: each admitted program applies the online
recurrence and its source-order padded stores to the current six-buffer
memory. Registers are absent from its inputs and discarded between programs. -/
def referenceSchedule (c : Call) (o : Ops α) : List (GridIndex c.grid) → State α → State α
  | [],s => s
  | idx :: rest,s =>
      let entry := Composition.prepare c.grid idx s
      let a := inputs entry
      referenceSchedule c o rest (scatter c o a (result c o a) entry)

theorem referenceSchedule_congr (c : Call) (o : Ops α) (jobs : List (GridIndex c.grid))
    (s t : State α) (memory : s.mem=t.mem) :
    (referenceSchedule c o jobs s).mem=(referenceSchedule c o jobs t).mem := by
  induction jobs generalizing s t with
  | nil => exact memory
  | cons idx rest ih =>
    simp only [referenceSchedule]
    have input : inputs (Composition.prepare c.grid idx s)=inputs (Composition.prepare c.grid idx t) := by
      simp only [Composition.prepare,inputs,memory]
    rw [input]
    apply ih
    exact scatter_mem_congr c o _ _ _ _ memory

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
    · change (exec (engine o.algebra) (Kernels.source c) (Composition.prepare c.grid idx s)).bind _=some final
      rw [hmid]
      exact hfinal
    · rw [meaning]
      exact referenceSchedule_congr c o rest _ _ effect

def gridResult (c : Call) (o : Ops α) (s : State α) : State α :=
  referenceSchedule c o (Composition.allPrograms c.grid) s

theorem complete_grid (c : Call) (o : Ops α) (positive : 0<c.BN) (s : State α) :
    ∃ final,Composition.schedule o.algebra (launched c) c.grid
      (Composition.allPrograms c.grid) s=some final ∧ final.mem=(gridResult c o s).mem :=
  complete_schedule c o positive _ s

#axiomsClean complete_schedule
#axiomsClean complete_grid
#specNonCircular gridResult avoiding [FP.Structural.exec,FP.Structural.run,Kernels.source,launched,complete_grid]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
