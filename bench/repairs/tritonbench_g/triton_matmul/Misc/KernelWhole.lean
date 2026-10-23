import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelTensor
import bench.repairs.tritonbench_g.triton_matmul.Misc.KernelFPEquiv

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelWhole
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter KernelContract
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- C's complete physical frame also preserves every byte of either original
input view. A and B may overlap each other and may have storage offsets. -/
theorem input_bytes (read bytes : Read) (a b c : Tensor) (output : Output a b c)
    (value : OutputIndex a b → Word) (post : ByteHost.Post read bytes a b c output.tensor value) :
    (∀ i j,i<a.metadata.numel → j<width (storage a.metadata) →
      bytes (a.metadata.dataPtr.toNat+width (storage a.metadata)*i+j)=
        read (a.metadata.dataPtr.toNat+width (storage a.metadata)*i+j)) ∧
    (∀ i j,i<b.metadata.numel → j<width (storage a.metadata) →
      bytes (b.metadata.dataPtr.toNat+width (storage a.metadata)*i+j)=
        read (b.metadata.dataPtr.toNat+width (storage a.metadata)*i+j)) :=
  ⟨ByteHost.separate_frame read bytes (storage a.metadata) a c output.separateA post.frame,
   ByteHost.separate_frame read bytes (storage a.metadata) b c output.separateB post.frame⟩

#specNonCircular KernelCorrect.expected avoiding
  [Source.typed,FP.Structural.exec,runPrograms,FiniteGrid.grid,CheckedHost.call,FiniteHost.call]
#specNonCircular KernelFPEquiv.expected avoiding
  [Source.typed,FP.Structural.exec,runPrograms,FiniteGrid.grid,ByteExecution.grid,ByteHost.call]
#axiomsClean KernelSource.executes
#axiomsClean KernelSource.safety
#axiomsClean source_store_unique
#axiomsClean KernelCorrect.kernel_real
#axiomsClean KernelTensor.real_tensor
#axiomsClean KernelNative.correct
#axiomsClean KernelFPEquiv.kernel_native_if_local
#axiomsClean input_bytes
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelWhole
