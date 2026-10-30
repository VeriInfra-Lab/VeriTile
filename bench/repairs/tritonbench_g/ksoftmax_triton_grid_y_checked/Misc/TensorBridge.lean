import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Correct
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Misc.Native
import bench.repairs.tritonbench_g.ksoftmax_triton.Correct
import bench.repairs.tritonbench_g.ksoftmax_triton.FPEquiv
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Contract

/-! Optional Tensor byte-frame application of the inherited native model. -/
namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax Ksoftmax.FPEquiv
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

theorem read_tensor_bytes_preserved (ops : Graph.Operations) (dst input : TensorMetadata)
    (separate : PairContract dst input) (before : Binding.Bytes) (s : State Graph.Expr)
    (j byte : Nat) (hj : j < input.numel) (hb : byte<4) :
    Binding.callerBytes ops dst before s (input.dataPtr.toNat+4*j+byte)=
      before (input.dataPtr.toNat+4*j+byte) :=
  Ksoftmax.FPEquiv.read_tensor_bytes_preserved ops dst input separate before s j byte hj hb

#axiomsClean read_tensor_bytes_preserved

end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.TensorBridge

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.TensorBridge.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax Ksoftmax.Correct
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-! Each endpoint derives actual live-axis support from candidate admission and
reuses the complete frozen mathematical source/Tensor/output/frame theorem. -/

theorem forward_kernel_real_words (depth : Host.DepthLibrary) (y x : TensorMetadata)
    (mask : ArgumentMetadata) (log : Bool) (mode : Option String) (causal : Bool)
    (admitted : KsoftmaxGridYChecked.ForwardContract y x mask mode)
    (xView : KernelTensor.View x) (maskView : KernelTensor.View (KernelInputs.maskTensor x mask mode))
    (memory : Nat → Graph.Word) (s : State (WithBot ℝ))
    (_inputs : ForwardWordDomain depth x mask mode causal xView maskView memory)
    (domain : ∀ (b : Fin (batches x)) (r : Fin (rows x)), 0<columns x →
      KernelExtended.valid (tensorForwardScores (depth.depth (columns x)) x mask mode causal xView maskView
        KernelIdeal.decode memory b r)) :
    SupportedLaunch x ∧ Spec.Real (∃ final,
      KernelExecution.exec Forward.TypedExecution.model depth (.forward log causal (Host.decodeMode mode)) x
        (KernelTensor.idealInitial KernelIdeal.decode y x (KernelInputs.maskTensor x mask mode) memory s)=some final ∧
      (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))), i.val<columns x →
        final.mem "dst" (Native.flat (rows x) (columns x) b r i.val)=Cell.mk .real
          (KernelExtended.formula
            (tensorForwardScores _ x mask mode causal xView maskView KernelIdeal.decode memory b r) log i)) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ x.numel≤j → final.mem region j=
        (KernelTensor.idealInitial KernelIdeal.decode y x (KernelInputs.maskTensor x mask mode) memory s).mem region j)) :=
  ⟨forward_launch admitted,Ksoftmax.Correct.forward_kernel_real_words depth y x mask log mode causal admitted.1 xView maskView memory s _inputs domain⟩

#axiomsClean forward_kernel_real_words

theorem backward_kernel_real_words (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata)
    (log causal : Bool) (admitted : KsoftmaxGridYChecked.BackwardContract dst grad saved)
    (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (memory : Nat → Graph.Word) (s : State (WithBot ℝ))
    (domain : BackwardWordDomain depth grad saved causal gradView savedView memory) :
    SupportedLaunch saved ∧ Spec.Real (∃ final,
      KernelExecution.exec Forward.TypedExecution.model depth (.backward log causal) saved
        (KernelTensor.idealInitial KernelIdeal.decode dst grad saved memory s)=some final ∧
      (∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))), i.val<columns saved →
        final.mem "dst" (Native.flat (rows saved) (columns saved) b r i.val)=Cell.mk .real
          (some (tensorBackwardFormula _ grad saved log causal gradView savedView memory b r i))) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ saved.numel≤j → final.mem region j=
        (KernelTensor.idealInitial KernelIdeal.decode dst grad saved memory s).mem region j)) :=
  ⟨backward_launch admitted,Ksoftmax.Correct.backward_kernel_real_words depth dst grad saved log causal admitted.1 gradView savedView memory s domain⟩

#axiomsClean backward_kernel_real_words

theorem log_backward_kernel_real_words (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata)
    (causal : Bool) (admitted : KsoftmaxGridYChecked.BackwardContract dst grad saved)
    (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (memory : Nat → Graph.Word) (s : State (WithBot ℝ))
    (domain : LogBackwardWordDomain depth grad saved causal gradView savedView memory) :
    SupportedLaunch saved ∧ Spec.Real (∃ final,
      KernelExecution.exec Forward.TypedExecution.model depth (.backward Bool.true causal) saved
        (KernelTensor.idealInitial KernelIdeal.decode dst grad saved memory s)=some final ∧
      (∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))), i.val<columns saved →
        final.mem "dst" (Native.flat (rows saved) (columns saved) b r i.val)=Cell.mk .real
          (some (tensorLogBackwardFormula _ grad saved causal gradView savedView memory b r i))) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ saved.numel≤j → final.mem region j=
        (KernelTensor.idealInitial KernelIdeal.decode dst grad saved memory s).mem region j)) :=
  ⟨backward_launch admitted,Ksoftmax.Correct.log_backward_kernel_real_words depth dst grad saved causal admitted.1 gradView savedView memory s domain⟩

#axiomsClean log_backward_kernel_real_words
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.TensorBridge.Correct

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.TensorBridge.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax Ksoftmax.FPEquiv
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-! All FP32 words, Tensor bindings, source execution, bytes and frames are
inherited without adding numerical or host-proof gates. Axis support is derived
from the exact candidate admission domain. -/

theorem forward_kernel_native (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (depth : Host.DepthLibrary) (y x : TensorMetadata) (mask : ArgumentMetadata)
    (log : Bool) (mode : Option String) (causal : Bool) (admitted : KsoftmaxGridYChecked.ForwardContract y x mask mode)
    (code : KernelCompiler.Selected platform (.forward log causal (Host.decodeMode mode))
      (rows x) (columns x) (depth.depth (columns x)) y x (KernelInputs.maskTensor x mask mode))
    (yView : KernelTensor.View y) (xView : KernelTensor.View x)
    (maskView : KernelTensor.View (KernelInputs.maskTensor x mask mode))
    (memory : Nat → Graph.Word) (s : State Graph.Expr) (before : Binding.Bytes)
    (backing : KernelTensor.Backed memory before [y,x,KernelInputs.maskTensor x mask mode]) :
    SupportedLaunch x ∧ KernelResult rest platform depth (.forward log causal (Host.decodeMode mode)) y x
      (KernelInputs.maskTensor x mask mode) x code yView xView maskView memory s before ∧
    (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))), i.val<columns x →
      (KernelAccess.matrixPointer y (rows x) (columns x) b r i).toNat=
        KernelTensor.address y yView (Native.flat (rows x) (columns x) b r i) ∧
      (KernelAccess.matrixPointer x (rows x) (columns x) b r i).toNat=
        KernelTensor.address x xView (Native.flat (rows x) (columns x) b r i)) ∧
    (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))),
      i.val<columns x → (Host.decodeMode mode).present=Bool.true →
      (KernelAccess.maskPointer (KernelInputs.maskTensor x mask mode) (Host.decodeMode mode) b r (columns x) i).toNat=
        KernelTensor.address (KernelInputs.maskTensor x mask mode) maskView
          (Forward.maskAddress (Host.decodeMode mode) b r (columns x) i)) :=
  ⟨forward_launch admitted,Ksoftmax.FPEquiv.forward_kernel_native rest platform depth y x mask log mode causal admitted.1 code yView xView maskView memory s before backing⟩

#axiomsClean forward_kernel_native

theorem backward_kernel_native (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata) (log causal : Bool)
    (admitted : KsoftmaxGridYChecked.BackwardContract dst grad saved)
    (code : KernelCompiler.Selected platform (.backward log causal)
      (rows saved) (columns saved) (depth.depth (columns saved)) dst grad saved)
    (dstView : KernelTensor.View dst) (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (memory : Nat → Graph.Word) (s : State Graph.Expr) (before : Binding.Bytes)
    (backing : KernelTensor.Backed memory before [dst,grad,saved]) :
    SupportedLaunch saved ∧ KernelResult rest platform depth (.backward log causal) dst grad saved saved code
      dstView gradView savedView memory s before ∧
    (∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))), i.val<columns saved →
      (KernelAccess.matrixPointer dst (rows saved) (columns saved) b r i).toNat=
        KernelTensor.address dst dstView (Native.flat (rows saved) (columns saved) b r i) ∧
      (KernelAccess.matrixPointer grad (rows saved) (columns saved) b r i).toNat=
        KernelTensor.address grad gradView (Native.flat (rows saved) (columns saved) b r i) ∧
      (KernelAccess.matrixPointer saved (rows saved) (columns saved) b r i).toNat=
        KernelTensor.address saved savedView (Native.flat (rows saved) (columns saved) b r i)) :=
  ⟨backward_launch admitted,Ksoftmax.FPEquiv.backward_kernel_native rest platform depth dst grad saved log causal admitted.1 code dstView gradView savedView memory s before backing⟩

#axiomsClean backward_kernel_native

theorem empty_kernel {α : Type} [Inhabited α] (model : Algebra α)
    (depth : Host.DepthLibrary) (op : Operation) (x : TensorMetadata)
    (s : State α) (empty : x.numel=0) :
    KernelExecution.exec model depth op x s=some s :=
  Ksoftmax.FPEquiv.empty_kernel model depth op x s empty

#axiomsClean empty_kernel

end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.TensorBridge.FPEquiv
