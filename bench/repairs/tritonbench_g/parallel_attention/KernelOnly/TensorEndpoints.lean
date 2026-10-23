import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.NativeCombined
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Tensor
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Coverage

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.TensorBinding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open NativeStage
noncomputable section
set_option maxHeartbeats 3000000

/-- Actual contiguous, resolved input Tensors and their admitted dense shapes.
Their storage offsets are retained by each Tensor's API binding. -/
structure ForwardTensorInputs (shape : Shape) where
  query : View
  key : View
  value : View
  queryShape : query.tensor.metadata.shape=[shape.B,shape.H,shape.T,shape.K]
  keyShape : key.tensor.metadata.shape=[shape.B,shape.H,shape.T,shape.K]
  valueShape : value.tensor.metadata.shape=[shape.B,shape.H,shape.T,shape.V]

structure BackwardTensorInputs (shape : Shape) where
  forward : ForwardTensorInputs shape
  outGrad : View
  normGrad : View
  outGradShape : outGrad.tensor.metadata.shape=[shape.B,shape.H,shape.T,shape.V]
  normGradShape : normGrad.tensor.metadata.shape=[shape.B,shape.H,shape.T]

def tensorWord (read : Nat → VeriTile.Torch.FP32.Word) (v : View) (off : Nat) : Word :=
  (VeriTile.Torch.FP32.logicalValue read v.tensor v.laws off).bits

def initialForward (read : Nat → VeriTile.Torch.FP32.Word) (inputs : ForwardTensorInputs shape)
    (before : State Word) : State Word :=
  {before with mem := fun R off =>
    if R="q" then .mk .real (tensorWord read inputs.query off)
    else if R="k" then .mk .real (tensorWord read inputs.key off)
    else if R="v" then .mk .real (tensorWord read inputs.value off)
    else before.mem R off}

/-- Entry reads come from the actual Tensor logical API, while all other
tagged regions retain the supplied incoming state. Masked loads never require
an input value outside its admitted Tensor shape. -/
def initial (read : Nat → VeriTile.Torch.FP32.Word) (inputs : BackwardTensorInputs shape) (before : State Word) : State Word :=
  {before with mem := fun R off =>
    if R="q" then .mk .real (tensorWord read inputs.forward.query off)
    else if R="k" then .mk .real (tensorWord read inputs.forward.key off)
    else if R="v" then .mk .real (tensorWord read inputs.forward.value off)
    else if R="do" then .mk .real (tensorWord read inputs.outGrad off)
    else if R="dz" then .mk .real (tensorWord read inputs.normGrad off)
    else before.mem R off}

/-- A no-scale compiler specialization has the ABI literal one. Scaled
specializations retain the actual incoming ABI scale; no sqrt identity is
asserted for its floating representation. Artifact recognition and local
compiler routing are the external boundary documented beside this packet. -/
def ScaleABI (d : NativePlans.Descriptor) (σ : ℝ) : Prop := d.shape.useScale=Bool.false → σ=1

/-- Raw kernel allocations: one key shard for O/Z/DV, and the retained
value-shard dimension for DQ/DK scratch. -/
def forwardShape (shape : Shape) (R : RegionName) : List Nat :=
  if R="o" then [1,shape.B,shape.H,shape.T,shape.V] else [1,shape.B,shape.H,shape.T]
def backwardShape (shape : Shape) (R : RegionName) : List Nat :=
  if R="dv" then [1,shape.B,shape.H,shape.T,shape.V]
  else [blocks shape.V shape.BV,shape.B,shape.H,shape.T,shape.K]

/-- Whole forward grid, direct Tensor logical readback and physical word
frame. The target is evaluated from the initial actual Tensor inputs; there
is no desired output or execution-success premise. The same final tagged
state observes both O and Z, including the single-writer Z guard. -/
theorem forward_tensor (actual reference : Scalars Word) (laws : PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (_accepted : ShapeContract d.shape) (σ : ℝ) (_scaleABI : ScaleABI d σ)
    (read : Nat → VeriTile.Torch.FP32.Word) (inputs : ForwardTensorInputs d.shape) (before : State Word)
    (out : View) (outputRegion : out.region∈(["o","z"] : List RegionName))
    (_outputShape : out.tensor.metadata.shape=forwardShape d.shape out.region)
    (binding : ∀ i,i<out.tensor.metadata.numel →
      ((initialForward read inputs before).mem out.region i).read .real=(read (out.base+4*i)).bits) :
    ∃ final,runPrograms (fun p s => forwardRun actual d σ (withProgram p s))
        (programs d.shape 128) (initialForward read inputs before)=some final ∧
      (∀ (p : Program d.shape 128) i,i<out.tensor.metadata.numel → forwardWrites d.shape p out.region i →
        (VeriTile.Torch.FP32.logicalValue (project read out final) out.tensor out.laws i).bits=
          (NativeStage.forwardGridTarget reference d σ p (initialForward read inputs before) out.region i).read .real) ∧
      (∀ R off,(∀ p∈programs d.shape 128,¬forwardWrites d.shape p R off) →
        final.mem R off=(initialForward read inputs before).mem R off) ∧
      (∀ address,(¬∃ i,i<out.tensor.metadata.numel ∧
        (∃ p : Program d.shape 128,forwardWrites d.shape p out.region i) ∧ address=out.base+4*i) →
        project read out final address=read address) := by
  obtain ⟨final,executed,_,reads,frame⟩ := NativeStage.forward_grid actual reference laws d σ (initialForward read inputs before)
  refine ⟨final,executed,?_,frame,?_⟩
  · intro p i inside written
    rw [logical_read read out final i inside,reads p (program_member _ _ p) _ _ written]
  · intro address outside
    apply project_frame read out (initialForward read inputs before) final binding
      (fun i => ∃ p : Program d.shape 128,forwardWrites d.shape p out.region i) _ address outside
    intro i missing
    exact frame out.region i (by intro p _ written; exact missing ⟨p,written⟩)

/-- Parent backward launch, preserving partial DQ/DK scratch shards and
complete DV, transported to any actual supplied output Tensor. -/
theorem backward_tensor (actual reference : Scalars Word) (laws : PrimitiveLaws actual reference)
    (d : NativePlans.Descriptor) (_accepted : ShapeContract d.shape) (σ : ℝ) (_scaleABI : ScaleABI d σ)
    (read : Nat → VeriTile.Torch.FP32.Word) (inputs : BackwardTensorInputs d.shape) (before : State Word)
    (out : View) (outputRegion : out.region∈(["dq","dk","dv"] : List RegionName))
    (_outputShape : out.tensor.metadata.shape=backwardShape d.shape out.region)
    (binding : ∀ i,i<out.tensor.metadata.numel →
      ((initial read inputs before).mem out.region i).read .real=(read (out.base+4*i)).bits) :
    ∃ final,runPrograms (combinedRun actual d σ) (programs d.shape 64) (initial read inputs before)=some final ∧
      (∀ (p : Program d.shape 64) i,i<out.tensor.metadata.numel → backwardWrites d.shape p out.region i →
        (VeriTile.Torch.FP32.logicalValue (project read out final) out.tensor out.laws i).bits=
          (NativeStage.backwardGridTarget reference d σ p (initial read inputs before) out.region i).read .real) ∧
      (∀ R off,(∀ p∈programs d.shape 64,¬backwardWrites d.shape p R off) →
        final.mem R off=(initial read inputs before).mem R off) ∧
      (∀ address,(¬∃ i,i<out.tensor.metadata.numel ∧
        (∃ p : Program d.shape 64,backwardWrites d.shape p out.region i) ∧ address=out.base+4*i) →
        project read out final address=read address) := by
  obtain ⟨final,executed,_,reads,frame⟩ := NativeStage.backward_grid actual reference laws d σ (initial read inputs before)
  refine ⟨final,executed,?_,frame,?_⟩
  · intro p i inside written
    rw [logical_read read out final i inside,reads p (program_member _ _ p) _ _ written]
  · intro address outside
    apply project_frame read out (initial read inputs before) final binding
      (fun i => ∃ p : Program d.shape 64,backwardWrites d.shape p out.region i) _ address outside
    intro i missing
    exact frame out.region i (by intro p _ written; exact missing ⟨p,written⟩)

#axiomsClean forward_tensor
#axiomsClean backward_tensor
#specNonCircular initial avoiding [runPrograms,FP.Structural.exec,forwardRun,NativeStage.combinedRun]
#specNonCircular initialForward avoiding [runPrograms,FP.Structural.exec,forwardRun,NativeStage.combinedRun]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.TensorBinding
