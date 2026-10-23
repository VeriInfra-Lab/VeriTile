import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.JointBacking

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open TensorBinding
noncomputable section
set_option maxHeartbeats 500000
variable {shape : Shape} {read : Memory} {before : State Word}

/-- All initial source input regions bind to the same physical reader used
for every output. Equal physical input addresses therefore have equal words,
even when two input Tensors alias. -/
theorem ForwardBacking.input_binding {inputs : ForwardTensorInputs shape} (b : ForwardBacking shape read inputs before) :
    InputBinding read (initialForward read inputs before) (forwardInputs inputs) := by
  intro v member i inside
  simp only [forwardInputs,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl
  · rw [b.queryRegion]
    change tensorWord read inputs.query i=(read (address inputs.query i)).bits
    exact logical_word read inputs.query i inside
  · rw [b.keyRegion]
    change tensorWord read inputs.key i=(read (address inputs.key i)).bits
    exact logical_word read inputs.key i inside
  · rw [b.valueRegion]
    change tensorWord read inputs.value i=(read (address inputs.value i)).bits
    exact logical_word read inputs.value i inside

theorem BackwardBacking.input_binding {inputs : BackwardTensorInputs shape} (b : BackwardBacking shape read inputs before) :
    InputBinding read (initial read inputs before) (backwardInputs inputs) := by
  intro v member i inside
  simp only [backwardInputs,forwardInputs,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with (rfl|rfl|rfl)|(rfl|rfl)
  · rw [b.queryRegion]
    change tensorWord read inputs.forward.query i=(read (address inputs.forward.query i)).bits
    exact logical_word read inputs.forward.query i inside
  · rw [b.keyRegion]
    change tensorWord read inputs.forward.key i=(read (address inputs.forward.key i)).bits
    exact logical_word read inputs.forward.key i inside
  · rw [b.valueRegion]
    change tensorWord read inputs.forward.value i=(read (address inputs.forward.value i)).bits
    exact logical_word read inputs.forward.value i inside
  · rw [b.outGradRegion]
    change tensorWord read inputs.outGrad i=(read (address inputs.outGrad i)).bits
    exact logical_word read inputs.outGrad i inside
  · rw [b.normGradRegion]
    change tensorWord read inputs.normGrad i=(read (address inputs.normGrad i)).bits
    exact logical_word read inputs.normGrad i inside

#axiomsClean ForwardBacking.input_binding
#axiomsClean BackwardBacking.input_binding

theorem ForwardBacking.input_regions {inputs : ForwardTensorInputs shape} (b : ForwardBacking shape read inputs before)
    (v : View) (member : v∈forwardInputs inputs) : v.region∈(["q","k","v","do","dz"] : List RegionName) := by
  simp only [forwardInputs,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl <;> simp [b.queryRegion,b.keyRegion,b.valueRegion]

theorem BackwardBacking.input_regions {inputs : BackwardTensorInputs shape} (b : BackwardBacking shape read inputs before)
    (v : View) (member : v∈backwardInputs inputs) : v.region∈(["q","k","v","do","dz"] : List RegionName) := by
  simp only [backwardInputs,forwardInputs,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with (rfl|rfl|rfl)|(rfl|rfl) <;>
    simp [b.queryRegion,b.keyRegion,b.valueRegion,b.outGradRegion,b.normGradRegion]

#axiomsClean ForwardBacking.input_regions
#axiomsClean BackwardBacking.input_regions
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
