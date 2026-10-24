import bench.repairs.tritonbench_g.softmax_flaggems.Kernels
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Host
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Contract
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Pointers

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.Correct
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section

/-- Contiguous admission makes the source's former contiguous conversion an
identity. Fresh allocation metadata remains an ordinary local Torch response. -/
def layouts (emptyLike : List Nat → List Nat → List Nat) : LayoutPrimitives :=
  ⟨fun _ strides => strides,emptyLike⟩

/-- Independent unshifted mathematical softmax and conventional VJP. The
reference includes saved-output dependence, metadata and the whole frame. -/
abbrev ResultSpec := Host.ResultSpec

specification softmax_autograd_real (x grad : VeriTile.Torch.TensorMetadata)
    (dim : Int) (dtype : Option VeriTile.Torch.DType)
    (admitted : ForwardContract x dim dtype) (gradient : GradientContract x grad)
    (sms : Nat) (waves : WavesAboveOne) (emptyLike : List Nat → List Nat → List Nat)
    (inputStrides : List Nat) (trials : List (Fin 6)) (choice : Fin 6)
    (initial : BlockState) (inputValues emptyOutput gradValues emptyGradient : Nat → ℝ) :
    Spec.Real (checkForward x dim dtype=.ok () ∧ checkGradient x grad=.ok () ∧
      ∃ (ax : Fin x.shape.length) (result : AutogradResult),
        normalizeDim x.shape dim=some ax ∧
        Host.invoke x.shape dim x.device.index sms waves (layouts emptyLike) inputStrides
          trials choice "x" "out" "grad" "in_grad" initial inputValues emptyOutput gradValues emptyGradient=.ok result ∧
        ResultSpec x.shape ax x.device.index (layouts emptyLike) inputStrides
          "x" "out" "grad" "in_grad" initial inputValues emptyOutput gradValues emptyGradient result) := by
  obtain ⟨ax,hnorm⟩ := admitted_axis x dim dtype admitted
  obtain ⟨result,he,hs⟩ := Host.whole_call_real x.shape dim ax hnorm x.device.index sms waves
    (layouts emptyLike) inputStrides trials choice "x" "out" "grad" "in_grad" initial
    inputValues emptyOutput gradValues emptyGradient (by decide) (by decide) (by decide)
  exact ⟨(checkForward_iff x dim dtype).mpr admitted,(checkGradient_iff x grad).mpr gradient,ax,result,hnorm,he,hs⟩

#specNonCircular ResultSpec avoiding [Host.invoke,Host.nonempty,Grid.backwardRun,Grid.tunedBackward]
#axiomsClean softmax_autograd_real
end
end VeriTile.Bench.Repairs.SoftmaxFlaggems.Correct
