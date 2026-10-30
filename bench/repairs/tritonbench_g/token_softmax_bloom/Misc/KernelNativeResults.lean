import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelCore
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Native
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelNativeResults
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
      KernelRewrite.candidateRun (SoftmaxTriton1.NativeGraph.algebra I V) p work (Native.entry before)=some after ∧
      (∀ e∈Grid.events p (Native.entry before) work,
        ((after p.output (Grid.offset p (Native.entry before) e.1 e.2)).read .fp32)
          (SoftmaxTriton1.NativeGraph.owner V p.width e.2.1.val)=Native.target I V p before e.1 e.2.1.val) ∧
      (∀ (r : RegionName) i,(r≠p.output ∨ ∀ e∈Grid.events p (Native.entry before) work,
        i≠Grid.offset p (Native.entry before) e.1 e.2) → after r i=Native.entry before r i) := by
  obtain ⟨after,run,candidate,values,frame⟩ := KernelCore.complete
    (SoftmaxTriton1.NativeGraph.algebra I V) p positive work (Native.entry before) separate unique
  simp only [KernelNative.model_profile] at run candidate values
  refine ⟨after,run,candidate,?_,frame⟩
  intro e he
  rw [values e he]
  exact Native.value_graph I V p before e.1 e.2.1

#specNonCircular Native.target avoiding [source,indexCommutedSource,Grid.run,outputValues]
#axiomsClean token_softmax_grid_fp32
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelNativeResults
