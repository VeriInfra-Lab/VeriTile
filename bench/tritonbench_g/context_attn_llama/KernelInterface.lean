import bench.tritonbench_g.context_attn_llama.FPReferenceGrid
import bench.tritonbench_g.context_attn_llama.NativeInstructions

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RawKernel
open VeriTile.Triton FP.Structural FPSuccess
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
variable {α : Type} [Inhabited α]

theorem tile_outside (c : Host.Call) (M : Algebra α) (dots : DotSupport c.layout M)
    (idx : GridIndex c.grid) (s : State α) (rg : RegionName) (off : Nat)
    (different : rg≠c.regions.out) : (FPReference.tile c M dots idx s).mem rg off=s.mem rg off := by
  apply scatter_frame
  intro i active same address
  exact different same

theorem reference_frame (c : Host.Call) (M : Algebra α) (dots : DotSupport c.layout M)
    (order : List (GridIndex c.grid)) (s : State α) (metadata : Metadata c s)
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out)
    (rg : RegionName) (off : Nat) (outside : ¬writes c order rg off) :
    (FPReference.grid c M dots order s).mem rg off=s.mem rg off := by
  induction order generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    have next : Metadata c (FPReference.tile c M dots idx s) := by
      refine ⟨?_,?_,?_⟩
      · intro b hb
        rw [tile_outside c M dots idx s _ _ separate.1]
        exact metadata.1 b hb
      · intro b hb
        rw [tile_outside c M dots idx s _ _ separate.2.1]
        exact metadata.2.1 b hb
      · intro b hb
        rw [tile_outside c M dots idx s _ _ separate.2.2]
        exact metadata.2.2 b hb
    rw [FPReference.grid,ih _ next (by
      rintro ⟨q,hq,i,active,same,address⟩
      exact outside ⟨q,List.mem_cons_of_mem _ hq,i,active,same,address⟩)]
    apply scatter_frame
    intro i active same address
    apply outside
    refine ⟨idx,by simp,i,(active_matches c s metadata idx i).mp active,same,?_⟩
    exact address.trans (offset_matches c s metadata idx i)

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RawKernel
