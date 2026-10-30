import bench.repairs.tritonbench_g.token_softmax_bloom.Misc.KernelNativeResults
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.KernelSource
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Native

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.KernelNativeResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
noncomputable section

/-- Original and integer-index-commuted source grids execute to one raw
state. Every active FP32 instruction-graph output and the complete frame follow
from raw input words, scalar parameters and store separation/uniqueness. -/
theorem token_softmax_grid_fp32 (I : Native.Instructions) (V : Nat)
    (p : Parameters) (positive : 0 < p.width) (work : List Grid.PID)
    (before : Grid.Memory Native.Word) (separate : Grid.Separate p)
    (unique : Grid.Unique p (Native.entry before) work) :
    ∃ after, Grid.run (SoftmaxTriton1.NativeGraph.algebra I V) p work (Native.entry before)=some after ∧
      KernelSource.candidateRun (SoftmaxTriton1.NativeGraph.algebra I V) p work (Native.entry before)=some after ∧
      (∀ e∈Grid.events p (Native.entry before) work,
        ((after p.output (Grid.offset p (Native.entry before) e.1 e.2)).read .fp32)
          (SoftmaxTriton1.NativeGraph.owner V p.width e.2.1.val)=Native.target I V p before e.1 e.2.1.val) ∧
      (∀ (r : RegionName) i,(r≠p.output ∨ ∀ e∈Grid.events p (Native.entry before) work,
        i≠Grid.offset p (Native.entry before) e.1 e.2) → after r i=Native.entry before r i) :=
  TokenSoftmaxBloom.KernelNativeResults.token_softmax_grid_fp32 I V p positive work before separate unique

#specNonCircular Native.target avoiding [source,KernelSource.indexCommutedSource,Grid.run,outputValues]
#axiomsClean token_softmax_grid_fp32
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.KernelNativeResults
