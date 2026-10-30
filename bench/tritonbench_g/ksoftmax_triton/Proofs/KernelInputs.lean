import bench.tritonbench_g.ksoftmax_triton.Proofs.KernelTensor
import bench.tritonbench_g.ksoftmax_triton.Proofs.Observed

namespace VeriTile.Bench.Repairs.Ksoftmax.KernelInputs
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-- The ignored-M branch passes X itself to the JIT. -/
def maskTensor (x : TensorMetadata) (mask : ArgumentMetadata) (mode : Option String) : TensorMetadata :=
  if mode=none then x else match mask with | .tensor m => m | .nonTensor => x

theorem mask_admitted (y x : TensorMetadata) (mask : ArgumentMetadata) (mode : Option String)
    (admitted : ForwardContract y x mask mode) : TensorContract (maskTensor x mask mode) := by
  by_cases h : mode=none
  · simp [maskTensor,h,admitted.2.1]
  · have hm := admitted.2.2.2.2.2.2.1
    cases mask with
    | nonTensor => simp [MaskContract,h] at hm
    | tensor m =>
      have hm' : TensorContract m ∧ m.device=x.device ∧
          m.shape=[if mode=some "qk" then rows x else batches x,columns x] := by
        simpa [MaskContract,h] using hm
      simpa [maskTensor,h] using hm'.1

theorem forward_words (depth : Host.DepthLibrary) (y x : TensorMetadata) (mask : ArgumentMetadata)
    (mode : Option String) (log causal : Bool) (memory : Nat → Graph.Word) (s : State Graph.Expr) :
    InputWords depth (.forward log causal (Host.decodeMode mode)) x
      (KernelTensor.nativeInitial y x (maskTensor x mask mode) memory s)
      (KernelTensor.words memory x) (KernelTensor.words memory (maskTensor x mask mode)) := by
  constructor
  · intro b r j _
    exact KernelTensor.native_left _ _ _ _ _ _
  · intro b r j _ _
    exact KernelTensor.native_right _ _ _ _ _ _

theorem backward_words (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata)
    (log causal : Bool) (memory : Nat → Graph.Word) (s : State Graph.Expr) :
    InputWords depth (.backward log causal) saved
      (KernelTensor.nativeInitial dst grad saved memory s)
      (KernelTensor.words memory grad) (KernelTensor.words memory saved) := by
  constructor
  · intro b r j _
    exact KernelTensor.native_left _ _ _ _ _ _
  · intro b r j _ _
    exact KernelTensor.native_right _ _ _ _ _ _

#specNonCircular maskTensor avoiding [forward,backward,runRows]
#axiomsClean mask_admitted
#axiomsClean forward_words
#axiomsClean backward_words
end
end VeriTile.Bench.Repairs.Ksoftmax.KernelInputs
