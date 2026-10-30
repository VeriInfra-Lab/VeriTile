import bench.repairs.tritonbench_g.token_softmax_bloom.Misc.KernelRealResults
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.KernelSource
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Real

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.KernelRealResults
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
        i≠Grid.offset p before e.1 e.2) → after r i=before r i) :=
  TokenSoftmaxBloom.KernelRealResults.token_softmax_grid_real p positive work before separate unique numeric cover

#specNonCircular Real.target avoiding [source,Grid.run,outputValues]
#axiomsClean token_softmax_grid_real
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.KernelRealResults
