import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Input

namespace VeriTile.Bench.Repairs.FastCeLoss.Checked
open Source Host
open CrossEntropy1.Host (Step)
noncomputable section

def checkedSourceSHA256 : String := "c848b2bb538b8b0800a069e564b2a796dcdf23f0b2688e0fc8b6f73a5ecf07ff"

/-- Exact Python int/float tags exclude bool, Tensor scalars and other objects.
Integer scalar arguments use the signed native ABI; Python floats retain their
finite binary64 payload, including either signed zero. -/
def ScalarContract : PythonScalar → Prop
  | .integer value => -(2^63 : Int) ≤ value ∧ value < 2^63
  | .floating bits => ∃ value, float64Value bits=some value

structure ForwardMetadata where
  logitsTensor : Bool
  labelsTensor : Bool
  logitsStrided : Bool
  labelsStrided : Bool
  logitsCUDA : Bool
  labelsCUDA : Bool
  capPythonScalar : Bool
  scalePythonScalar : Bool
  itemsPythonScalar : Bool
  currentDevice : Nat

structure BackwardMetadata where
  tensor : Bool
  strided : Bool
  cuda : Bool
  currentDevice : Nat

structure ForwardContract (metadata : ForwardMetadata) (args : Arguments)
    (integers : Address → Int) : Prop where
  logitsTensor : metadata.logitsTensor=Bool.true
  labelsTensor : metadata.labelsTensor=Bool.true
  logitsStrided : metadata.logitsStrided=Bool.true
  labelsStrided : metadata.labelsStrided=Bool.true
  logitsCUDA : metadata.logitsCUDA=Bool.true
  labelsCUDA : metadata.labelsCUDA=Bool.true
  device : args.labels.device=args.logits.device
  current : args.logits.device=metadata.currentDevice
  logitsShape : args.logits.tensor.shape=[args.batch,args.sequence,args.vocab]
  labelsShape : args.labels.tensor.shape=[args.batch,args.sequence]
  logitsDType : args.logits.dtype=.fp32
  labelsDType : args.labels.dtype=.int64
  logitsContiguous : args.logits.contiguous=Bool.true
  labelsContiguous : args.labels.contiguous=Bool.true
  logitsResolved : args.logits.resolved
  labelsResolved : args.labels.resolved
  distinct : args.logits.tensor.storage≠args.labels.tensor.storage
  positive : 0<args.vocab
  rowGrid : args.shape.rows≤2^31-1
  columnGrid : Source.ceilDiv args.vocab 4096≤65535
  logitAlignment : 0<args.shape.rows → args.logits.pointer.toNat%4=0
  labelAlignment : 0<args.shape.rows → args.labels.pointer.toNat%8=0
  capType : metadata.capPythonScalar=Bool.true
  scaleType : metadata.scalePythonScalar=Bool.true
  cap : ScalarContract args.capArgument
  scale : ScalarContract args.scaleArgument
  labelsValid : ∀ row<args.shape.rows,
    integers (addr args.labels.tensor row)=-100 ∨
    (0 ≤ integers (addr args.labels.tensor row) ∧ integers (addr args.labels.tensor row)<args.vocab)
  denominator : match args.items with
    | none => 0<Source.count args.shape (fun row => integers (addr args.labels.tensor row))
    | some value => metadata.itemsPythonScalar=Bool.true ∧ ScalarContract value ∧
        ∃ real, pythonValue value=some real ∧ 0<real

structure BackwardContract (metadata : BackwardMetadata) (args : Arguments)
    (upstream : Descriptor) : Prop where
  tensor : metadata.tensor=Bool.true
  strided : metadata.strided=Bool.true
  cuda : metadata.cuda=Bool.true
  dtype : upstream.dtype=.fp32
  device : upstream.device=args.logits.device
  current : args.logits.device=metadata.currentDevice
  shape : upstream.tensor.shape=[]
  resolved : upstream.resolved
  alignment : upstream.pointer.toNat%4=0
  distinct : upstream.tensor.storage≠args.labels.tensor.storage

/-- Ordinary Tensor/allocator observations and scalar decoding, separated from
the task's executable guards. These contain no output or numerical equality. -/
structure Bindings {σ : Type} (live : σ → Nat → Prop) (args : Arguments)
    (upstream : Descriptor) (native : σ) : Prop where
  logitsBounds : args.logits.bounded
  labelsBounds : args.labels.bounded
  upstreamBounds : upstream.bounded
  logitsLive : live native args.logits.tensor.storage
  labelsLive : live native args.labels.tensor.storage
  upstreamLive : live native upstream.tensor.storage
  capBinding : pythonValue args.capArgument=some args.options.softcap
  scaleBinding : pythonValue args.scaleArgument=some args.options.scaling

/-- The guard runs before the unchanged public body; the scalar hook runs
after forward succeeds and before sum/division autograd and custom backward. -/
def whole {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (fm : ForwardMetadata) (bm : BackwardMetadata)
    (args : Arguments) (upstream : Descriptor) : Action α σ WholeReturn := fun initial => by
  classical
  exact if ForwardContract fm args initial.integer then
    bindAction (Host.forward a scalars runtime stream args)
      (fun saved middle => if BackwardContract bm args upstream then
        Host.backward a scalars runtime stream saved upstream middle
        else .error "checked scalar upstream" middle) initial
    else .error "checked forward contract" initial

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Checked
