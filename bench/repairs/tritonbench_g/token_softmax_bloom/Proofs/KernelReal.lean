import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelGeometry
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Real

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open KernelTensor KernelGrid
noncomputable section
set_option maxHeartbeats 5000000

abbrev Scalar := VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real.Scalar
abbrev model := VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real.algebra

def reads (read : Nat → ℝ) : Nat → Scalar := fun address => some (read address)

/-- Independent unshifted softmax on the original Tensor logical coordinates. -/
def reference (c : TensorCall) (metadataRead : Nat → Word) (read : Nat → ℝ)
    (b head n : Nat) : ℝ :=
  Real.exp (logicalValue read c.input head (start c metadataRead b+n)) /
    ∑ j ∈ Finset.range (length c metadataRead b),
      Real.exp (logicalValue read c.input head (start c metadataRead b+j))

@[simp] theorem model_profile : FP.ProfiledRewrite.engine model=model := rfl

theorem numeric (c : TensorCall) (metadataRead : Nat → Word) (read : Nat → ℝ)
    (before : NativeCells Scalar) (b head : Nat) :
    VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real.NumericInput (params c)
      (Grid.state (view c (reads read) metadataRead before) (b,head)) := by
  intro n hn
  simp [params,Grid.state,view,reads]

theorem input_logical (c : TensorCall) (metadataRead : Nat → Word) (read : Nat → ℝ)
    (before : NativeCells Scalar) (binding : InputBinding c (reads read) before)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (b head n : Nat) (hb : b < rows c.start.observation) (hh : head < rows c.input.observation)
    (hn : n < length c metadataRead b) :
    VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real.input (params c)
      (Grid.state (view c (reads read) metadataRead before) (b,head)) n=
        logicalValue read c.input head (start c metadataRead b+n) := by
  have linked := KernelBinding.source_native_load c (reads read) metadataRead before binding contract launch
    b head n hb hh hn
  have result := congrArg (WithBot.unbotD 0) (linked.1.trans linked.2.1)
  simpa [VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real.input,
    params,Grid.state,view,reads,logicalValue,startIndex,start,observedStart] using result

theorem target_logical (c : TensorCall) (metadataRead : Nat → Word) (read : Nat → ℝ)
    (before : NativeCells Scalar) (binding : InputBinding c (reads read) before)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (b head n : Nat) (hb : b < rows c.start.observation) (hh : head < rows c.input.observation)
    (hn : n < length c metadataRead b) :
    VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real.target (params c)
      (Grid.state (view c (reads read) metadataRead before) (b,head)) n=
        reference c metadataRead read b head n := by
  unfold VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Real.target reference
  rw [count_view,input_logical c metadataRead read before binding contract launch b head n hb hh hn]
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  rw [input_logical c metadataRead read before binding contract launch b head j hb hh (Finset.mem_range.mp hj)]

#axiomsClean numeric
#axiomsClean target_logical
#specNonCircular reference avoiding [FP.Structural.exec,Grid.run,outputValues]
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelReal
