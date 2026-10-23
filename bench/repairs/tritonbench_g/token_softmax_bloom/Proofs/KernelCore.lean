import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelRewrite
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Stores

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelCore
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
noncomputable section

/-- Shared raw-grid execution theorem. Separation and uniqueness concern only
source regions and active store addresses, independent of Tensor admission. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (positive : 0 < p.width) (work : List Grid.PID) (before : Grid.Memory α)
    (separate : Grid.Separate p) (unique : Grid.Unique p before work) :
    ∃ after, Grid.run (FP.ProfiledRewrite.engine M) p work before=some after ∧
      KernelRewrite.candidateRun (FP.ProfiledRewrite.engine M) p work before=some after ∧
      (∀ e∈Grid.events p before work,
        after p.output (Grid.offset p before e.1 e.2)=
          .mk .fp32 (Grid.values (FP.ProfiledRewrite.engine M) p before e)) ∧
      (∀ (r : RegionName) i,(r≠p.output ∨ ∀ e∈Grid.events p before work,
        i≠Grid.offset p before e.1 e.2) → after r i=before r i) := by
  obtain ⟨after,run,_,values,frame⟩ := Grid.whole (FP.ProfiledRewrite.engine M) p positive work before separate unique
  exact ⟨after,run,(KernelRewrite.grid_execution M p work before).symm.trans run,values,frame⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelCore
