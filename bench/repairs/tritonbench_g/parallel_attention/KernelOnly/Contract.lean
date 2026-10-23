import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.TensorEndpoints

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Contract
export VeriTile.Bench.Repairs.ParallelAttention (ShapeContract)
export VeriTile.Bench.Repairs.ParallelAttentionKernel.TensorBinding (ScaleABI)
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Contract

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open TensorBinding
noncomputable section

abbrev Word := NativeStage.Word
abbrev Memory := Nat → VeriTile.Torch.FP32.Word

def address (v : View) (i : Nat) : Nat := v.base+4*i
def inStorage (v : View) (a : Nat) : Prop :=
  v.tensor.storage.base≤a ∧ a<v.tensor.storage.base+4*v.tensor.storage.elements

inductive ForwardPort where | output | normalizer deriving DecidableEq
inductive BackwardPort where | query | key | value deriving DecidableEq

def ForwardPort.region : ForwardPort → RegionName
  | .output => "o"
  | .normalizer => "z"
def BackwardPort.region : BackwardPort → RegionName
  | .query => "dq"
  | .key => "dk"
  | .value => "dv"

/-- Spatial binding of all simultaneous output ports. The initial values
come from one physical reader. Separation concerns addresses, not desired
results, arithmetic, or any completed execution. -/
structure Outputs (κ : Type) (regions : κ → RegionName) (shapes : κ → List Nat)
    (read : Memory) (source : State Word) where
  view : κ → View
  region : ∀ k,(view k).region=regions k
  shape : ∀ k,(view k).tensor.metadata.shape=shapes k
  initial : ∀ k i,i<(view k).tensor.metadata.numel →
    (source.mem (view k).region i).read .real=(read (address (view k) i)).bits
  separate : ∀ k l,k≠l → ∀ i j,i<(view k).tensor.metadata.numel →
    j<(view l).tensor.metadata.numel → address (view k) i≠address (view l) j

def forwardInputs (inputs : ForwardTensorInputs shape) : List View :=
  [inputs.query,inputs.key,inputs.value]
def backwardInputs (inputs : BackwardTensorInputs shape) : List View :=
  forwardInputs inputs.forward ++ [inputs.outGrad,inputs.normGrad]

/-- Output words are separate from the complete backing allocations of
the input Tensors. Input Tensors may alias each other, share storage, and
have nonzero storage offsets. No allocator execution is assumed. -/
def InputSeparation (outs : κ → View) (ins : List View) : Prop :=
  ∀ k i,i<(outs k).tensor.metadata.numel → ∀ v∈ins,¬inStorage v (address (outs k) i)

def InputBinding (read : Memory) (source : State Word) (ins : List View) : Prop :=
  ∀ v∈ins,∀ i,i<v.tensor.metadata.numel →
    (source.mem v.region i).read .real=(read (address v i)).bits

structure ForwardBacking (shape : Shape) (read : Memory) (inputs : ForwardTensorInputs shape)
    (before : State Word) where
  outputs : Outputs ForwardPort ForwardPort.region (fun k => forwardShape shape k.region)
    read (initialForward read inputs before)
  queryRegion : inputs.query.region="q"
  keyRegion : inputs.key.region="k"
  valueRegion : inputs.value.region="v"
  inputSeparate : InputSeparation outputs.view (forwardInputs inputs)

structure BackwardBacking (shape : Shape) (read : Memory) (inputs : BackwardTensorInputs shape)
    (before : State Word) where
  outputs : Outputs BackwardPort BackwardPort.region (fun k => backwardShape shape k.region)
    read (initial read inputs before)
  queryRegion : inputs.forward.query.region="q"
  keyRegion : inputs.forward.key.region="k"
  valueRegion : inputs.forward.value.region="v"
  outGradRegion : inputs.outGrad.region="do"
  normGradRegion : inputs.normGrad.region="dz"
  inputSeparate : InputSeparation outputs.view (backwardInputs inputs)

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
