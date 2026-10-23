import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeImages
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.PhysicalBindings

namespace VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Physical
open CrossEntropy1.Host (World warps)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- The actual ordered Python ABI pointer list. Kernel outputs already exist
at the call boundary; an allocator's output descriptor is observed afterward. -/
structure PointerArgument (σ : Type) where
  role : String
  tensor : Tensor
  state : σ

def SourceCall.pointers {σ : Type} (call : SourceCall σ) : List (PointerArgument σ) :=
  match call.kind,call.inputs,call.outputs with
  | .forward,[logits,labels],[loss,lse,zloss] =>
      [⟨"loss_ptr",loss,call.before.native⟩,⟨"lse_ptr",lse,call.before.native⟩,
       ⟨"z_loss_ptr",zloss,call.before.native⟩,⟨"logits_ptr",logits,call.before.native⟩,
       ⟨"labels_ptr",labels,call.before.native⟩]
  | .backward,[logits,labels,lse,upstream],[gradient] =>
      [⟨"dlogits_ptr",gradient,call.before.native⟩,⟨"dloss_ptr",upstream,call.before.native⟩,
       ⟨"logits_ptr",logits,call.before.native⟩,⟨"lse_ptr",lse,call.before.native⟩,
       ⟨"labels_ptr",labels,call.before.native⟩]
  | _,_,_ => call.inputs.map (fun t => ⟨"input",t,call.before.native⟩) ++
      call.outputs.map (fun t => ⟨"output",t,call.after.native⟩)

def elementBytes (dtype : String) : Nat := if dtype="torch.int64" then 8 else if dtype="torch.bool" then 1 else 4

def PointerArgument.metadata {σ : Type} (layout : Layout σ) (allocation : σ → RegionName → Nat)
    (arg : PointerArgument σ) : InvocationGraphs.PointerMetadata where
  role := arg.role
  dtype := arg.tensor.dtype
  address := (layout.buffer arg.state arg.tensor.region).base+elementBytes arg.tensor.dtype*arg.tensor.storageOffset
  allocation := allocation arg.state arg.tensor.region
  storageOffsetBytes := elementBytes arg.tensor.dtype*arg.tensor.storageOffset
  shape := arg.tensor.shape
  strides := arg.tensor.strides
  device := (layout.buffer arg.state arg.tensor.region).device
  isNegativeView := arg.tensor.isNeg

def requestKind : CallKind → InvocationGraphs.Kind
  | .forward => .forward | .backward => .backward | .torchSum => .localSum | .torchLSE => .localLSE
  | .square => .square | .scale => .scale | .add => .add | .reduceWait => .rankSum
  | .mask => .runtime "masked_fill_" | .resolve => .runtime "resolve_neg" | .clone => .runtime "clone"
  | .contiguous => .runtime "contiguous" | .copy => .runtime "copy_" | .gather => .runtime "all_gather"
  | .reduceStart => .runtime "all_reduce_async" | .query => .runtime "group_query"
  | .allocate => .runtime "allocation" | .context => .runtime "device_context" | .setup => .runtime "setup"
  | .save => .runtime "autograd_metadata" | .compare => .runtime "equal_int"

def SourceCall.scalarWords {σ : Type} (encode : ℝ → CrossEntropy1.Floating.Word) (call : SourceCall σ) : List CrossEntropy1.Floating.Word :=
  match call.arguments with
  | .kernel p _ _ => [encode p.smoothing,encode p.logitScale,encode p.squareScale]
  | .scalar x => [encode x]
  | _ => []

def SourceCall.scalarIntegers {σ : Type} (call : SourceCall σ) : List Int :=
  match call.arguments with
  | .kernel p _ _ => [p.ignored,p.totalClasses,p.classStart,p.cols] ++
      if call.kind=.forward then [p.rows,p.logitsStride].map Int.ofNat else [p.logitsStride,p.gradientStride,p.upstreamStride].map Int.ofNat
  | .natural x => [x]
  | .integer x => [x]
  | _ => []

def SourceCall.constexpr {σ : Type} (call : SourceCall σ) : List (String × Int) :=
  match call.arguments with
  | .kernel p _ _ => [("BLOCK_SIZE",(p.block : Int)),("HAS_SMOOTHING",if p.smoothing>0 then 1 else 0)] ++
      if call.kind=.forward then [("SPLIT",if p.split then 1 else 0)] else []
  | _ => []

/-- Nonnumerical full-request binding. The encoding argument is the ordinary
scalar ABI conversion, not an equality about a task result. Compiler version,
target, stages and fusion remain in the request consumed by the actual loader. -/
structure RequestMatches {σ : Type} (layout : Layout σ) (allocation : σ → RegionName → Nat)
    (encode : ℝ → CrossEntropy1.Floating.Word) (call : SourceCall σ) (request : InvocationGraphs.Request σ) : Prop where
  before : request.before=call.before.native
  source : request.sourceDigest="b690a712bd968686ee78818e81dbbb21a8d15331e132eb1c994d1ed3eb2b4ba5"
  kind : request.kind=requestKind call.kind
  pointers : request.pointers=call.pointers.map (PointerArgument.metadata layout allocation)
  scalars : request.scalarWords=call.scalarWords encode
  integers : request.scalarIntegers=call.scalarIntegers
  constants : request.constexpr=call.constexpr
  kernel : ∀ p count warps, call.arguments=.kernel p count warps →
    request.grid=[p.rows,count] ∧ request.blockWidth=p.block ∧ request.warps=warps
  group : request.group=if call.kind=.gather ∨ call.kind=.reduceStart ∨ call.kind=.reduceWait then some call.group else none
  work : request.work=call.work

/-- Exact setup witnesses, extracted from these successful Ops launches.
The grid's memory transformation is the actual typed source interpreter. -/
def SourceCall.Setup {σ : Type} (call : SourceCall σ) : Prop :=
  ∀ p count n, call.arguments=.kernel p count n → ∃ entered ready : World Expr σ,
    call.runtime.enterDevice call.device call.before=.ok () entered ∧
    (if call.kind=.forward then call.runtime.setupForwardOps p n entered else call.runtime.setupBackwardOps p n entered)=.ok () ready ∧
    call.runtime.exitDevice call.device
      {ready with memory:=if call.kind=.forward then Grid.forward algebra p ready.memory count else Grid.backward algebra p ready.memory count}=.ok () call.after

theorem forward_setup {σ : Type} {obs} (call : Entry.Invocation (σ:=σ) obs algebra) :
    (SourceCall.forward call).Setup := by
  intro p count n same
  have equal : call.data.parameters=p ∧ call.data.count=count ∧ warps call.data.parameters.block=n := by
    simpa only [SourceCall.forward,CallArguments.kernel.injEq] using same
  rcases equal with ⟨rfl,rfl,rfl⟩
  exact ⟨call.kernel.launch.entered,call.kernel.launch.ready,call.kernel.launch.enterCall,
    call.kernel.launch.setupCall,call.kernel.launch.exitCall⟩

theorem backward_setup {σ : Type} {rt : Runtime Expr σ} {c input output before after}
    (b : BackwardData algebra rt c input output before after) : (SourceCall.backward b).Setup := by
  intro p count n same
  have equal : b.parameters=p ∧ b.count=count ∧ warps b.parameters.block=n := by
    simpa only [SourceCall.backward,CallArguments.kernel.injEq] using same
  rcases equal with ⟨rfl,rfl,rfl⟩
  let launch := Classical.choice (backward_launch_success algebra rt c.logits.device b.parameters b.count b.prepared b.computed b.launchCall)
  exact ⟨launch.entered,launch.ready,launch.enterCall,launch.setupCall,launch.exitCall⟩

/-- Ordinary device-entry/setup metadata stability for the actual pointers.
This connects the compiler specialization addresses at the wrapper boundary
to those of the kernel's ready state; it is separate from numerical ports. -/
def SourceCall.ReadyPointers {σ : Type} (layout : Layout σ) (call : SourceCall σ) : Prop :=
  ∀ p count n, call.arguments=.kernel p count n → ∃ entered ready : World Expr σ,
    call.runtime.enterDevice call.device call.before=.ok () entered ∧
    (if call.kind=.forward then call.runtime.setupForwardOps p n entered else call.runtime.setupBackwardOps p n entered)=.ok () ready ∧
    (∀ arg∈call.pointers, layout.buffer ready.native arg.tensor.region=layout.buffer arg.state arg.tensor.region)

/-- The same request selects the object and every source-occurrence fusion
choice. Descriptor identity, stream and rank are checked at the actual slot. -/
structure RequestBinding {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (layout : Layout σ)
    (allocation : σ → RegionName → Nat) (encode : ℝ → CrossEntropy1.Floating.Word)
    (fusion : ConfiguredPlan.FusionSelection σ) (policy : ConfiguredPlan.Policy fusion) : Prop where
  actual : ∀ d (called : selection.invoked d),
    RequestMatches layout allocation encode (objects.exact d called).call
      (abi.request (objects.exact d called).location (objects.exact d called).call)
  setup : ∀ d (called : selection.invoked d), (objects.exact d called).call.Setup
  readyPointers : ∀ d (called : selection.invoked d), (objects.exact d called).call.ReadyPointers layout
  request : ∀ d (called : selection.invoked d), policy.request d=
    abi.request (objects.exact d called).location (objects.exact d called).call
  implementation : ∀ d (called : selection.invoked d), (policy.request d).implementation=d.implementation

#axiomsClean forward_setup
#axiomsClean backward_setup
#specNonCircular RequestMatches avoiding [Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
