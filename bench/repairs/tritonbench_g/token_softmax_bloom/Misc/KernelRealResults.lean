import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelCore
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Real
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelRealResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
noncomputable section

/-- Independent softmax values for every active output in a complete raw
source schedule, with successful execution and the entire exterior Cell frame. -/
theorem token_softmax_grid_real (p : Parameters) (positive : 0 < p.width)
    (work : List Grid.PID) (before : Grid.Memory Real.Scalar)
    (separate : Grid.Separate p) (unique : Grid.Unique p before work)
    (numeric : ∀ pid∈work,Real.NumericInput p (Grid.state before pid))
    (cover : ∀ pid∈work,count p (Grid.state before pid) ≤ p.width) :
    ∃ after, Grid.run Real.algebra p work before=some after ∧
      (∀ e∈Grid.events p before work,
        WithBot.unbotD 0 ((after p.output (Grid.offset p before e.1 e.2)).read .fp32)=
          Real.target p (Grid.state before e.1) e.2.1.val) ∧
      (∀ (r : RegionName) i,(r≠p.output ∨ ∀ e∈Grid.events p before work,
        i≠Grid.offset p before e.1 e.2) → after r i=before r i) := by
  obtain ⟨after,run,_,values,frame⟩ := KernelCore.complete Real.algebra p positive work before separate unique
  simp only [KernelReal.model_profile] at run values
  refine ⟨after,run,?_,frame⟩
  intro e he
  have active := (Grid.mem_events p before work e).mp he
  rw [values e he]
  exact Real.output_target p (Grid.state before e.1) positive (numeric e.1 active.1)
    (cover e.1 active.1) e.2.1 active.2

#specNonCircular Real.target avoiding [source,Grid.run,outputValues]
#axiomsClean token_softmax_grid_real
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelRealResults
