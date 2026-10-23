import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.JointCoverage
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.JointInputs

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open TensorBinding
noncomputable section
set_option maxHeartbeats 500000

/-- One successful source grid produces one physical post for O and Z
simultaneously. Every Tensor logical observation, input-storage frame, and
union frame in this conclusion refers to that same post. -/
theorem forward_joint (actual reference : Scalars Word) (laws : NativeStage.PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (_accepted : ShapeContract d.shape) (σ : ℝ) (_scaleABI : ScaleABI d σ)
    (read : Memory) (inputs : ForwardTensorInputs d.shape) (before : State Word)
    (backing : ForwardBacking d.shape read inputs before) :
    ∃ final post,
      runPrograms (fun p s => NativeStage.forwardRun actual d σ (withProgram p s))
        (programs d.shape 128) (initialForward read inputs before)=some final ∧
      post=jointPost read backing.outputs.view (forwardFootprint d.shape) final ∧
      JointBinding read post (initialForward read inputs before) final backing.outputs.view
        (forwardInputs inputs) (forwardFootprint d.shape) ∧
      (∀ (p : Program d.shape 128) (k : ForwardPort) i,
        forwardWrites d.shape p (backing.outputs.view k).region i →
        (VeriTile.Torch.FP32.logicalValue post (backing.outputs.view k).tensor (backing.outputs.view k).laws i).bits=
          (NativeStage.forwardGridTarget reference d σ p (initialForward read inputs before)
            (backing.outputs.view k).region i).read .real) ∧
      (∀ R i,¬forwardFootprint d.shape R i → final.mem R i=(initialForward read inputs before).mem R i) ∧
      (∀ (p q : Program d.shape 128) (k l : ForwardPort) i j,
        forwardWrites d.shape p (backing.outputs.view k).region i →
        forwardWrites d.shape q (backing.outputs.view l).region j →
        address (backing.outputs.view k) i=address (backing.outputs.view l) j → p=q ∧ k=l ∧ i=j) := by
  obtain ⟨final,executed,inputsSame,reads,frame⟩ := NativeStage.forward_grid actual reference laws d σ
    (initialForward read inputs before)
  have sourceFrame : ∀ R i,¬forwardFootprint d.shape R i → final.mem R i=(initialForward read inputs before).mem R i := by
    intro R i missing
    exact frame R i (by intro p _ written; exact missing ⟨p,written⟩)
  have joint := realize_joint backing.outputs (forwardInputs inputs) backing.input_binding
    backing.inputSeparate (forwardFootprint d.shape) final sourceFrame (by
      intro v member
      exact (inputsSame v.region (backing.input_regions v member)).symm)
  refine ⟨final,jointPost read backing.outputs.view (forwardFootprint d.shape) final,
    executed,rfl,joint,?_,sourceFrame,?_⟩
  · intro p k i written
    rw [joint.logicalOutputs k i (forward_store_inside backing.outputs p k i written),
      reads p (program_member _ _ p) _ _ written]
  · exact forward_physical_unique backing.outputs

/-- The actual parent DQ/barrier/DKV grid produces one common physical
post for DQ, DK and DV. DQ/DK keep their value-shard scratch dimension. -/
theorem backward_joint (actual reference : Scalars Word) (laws : NativeStage.PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (_accepted : ShapeContract d.shape) (σ : ℝ) (_scaleABI : ScaleABI d σ)
    (read : Memory) (inputs : BackwardTensorInputs d.shape) (before : State Word)
    (backing : BackwardBacking d.shape read inputs before) :
    ∃ final post,
      runPrograms (NativeStage.combinedRun actual d σ) (programs d.shape 64)
        (initial read inputs before)=some final ∧
      post=jointPost read backing.outputs.view (backwardFootprint d.shape) final ∧
      JointBinding read post (initial read inputs before) final backing.outputs.view
        (backwardInputs inputs) (backwardFootprint d.shape) ∧
      (∀ (p : Program d.shape 64) (k : BackwardPort) i,
        backwardWrites d.shape p (backing.outputs.view k).region i →
        (VeriTile.Torch.FP32.logicalValue post (backing.outputs.view k).tensor (backing.outputs.view k).laws i).bits=
          (NativeStage.backwardGridTarget reference d σ p (initial read inputs before)
            (backing.outputs.view k).region i).read .real) ∧
      (∀ R i,¬backwardFootprint d.shape R i → final.mem R i=(initial read inputs before).mem R i) ∧
      (∀ (p q : Program d.shape 64) (k l : BackwardPort) i j,
        backwardWrites d.shape p (backing.outputs.view k).region i →
        backwardWrites d.shape q (backing.outputs.view l).region j →
        address (backing.outputs.view k) i=address (backing.outputs.view l) j → p=q ∧ k=l ∧ i=j) := by
  obtain ⟨final,executed,inputsSame,reads,frame⟩ := NativeStage.backward_grid actual reference laws d σ
    (initial read inputs before)
  have sourceFrame : ∀ R i,¬backwardFootprint d.shape R i → final.mem R i=(initial read inputs before).mem R i := by
    intro R i missing
    exact frame R i (by intro p _ written; exact missing ⟨p,written⟩)
  have joint := realize_joint backing.outputs (backwardInputs inputs) backing.input_binding
    backing.inputSeparate (backwardFootprint d.shape) final sourceFrame (by
      intro v member
      exact (inputsSame v.region (backing.input_regions v member)).symm)
  refine ⟨final,jointPost read backing.outputs.view (backwardFootprint d.shape) final,
    executed,rfl,joint,?_,sourceFrame,?_⟩
  · intro p k i written
    rw [joint.logicalOutputs k i (backward_store_inside backing.outputs p k i written),
      reads p (program_member _ _ p) _ _ written]
  · exact backward_physical_unique backing.outputs

#axiomsClean forward_joint
#axiomsClean backward_joint
#specNonCircular forwardFootprint avoiding [jointPost,FP.Structural.exec,runPrograms,NativeStage.forwardRun]
#specNonCircular backwardFootprint avoiding [jointPost,FP.Structural.exec,runPrograms,NativeStage.combinedRun]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
