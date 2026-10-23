import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeSlots
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ConfiguredPlan
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.JointExecution
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NativeInputs

namespace VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Physical
open CrossEntropy1.Host (World)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Stream and chosen launch configuration are ordinary runtime metadata for
this full source call. `loaded` below also receives that complete source call. -/
structure InvocationABI (σ : Type) where
  request : SourceLocation → SourceCall σ → InvocationGraphs.Request σ
  identity : SourceLocation → Nat
  injective : Function.Injective identity
  rank : SourceLocation → Nat
  peerParticipant : Nat → Participant
  stream : SourceLocation → SourceCall σ → Nat
  configuration : SourceLocation → SourceCall σ → List Nat → Prop

structure DescriptorMatches {σ : Type} (abi : InvocationABI σ) (location : SourceLocation)
    (source : SourceCall σ) (d : CallDescriptor σ) : Prop where
  invocation : d.invocation=abi.identity location
  before : d.before=source.before.native
  after : d.after=source.after.native
  inputs : d.inputs=source.inputs
  outputs : d.outputs=source.outputs
  device : d.device=source.device
  stream : d.stream=abi.stream location source
  rank : d.rank=abi.rank location
  group : d.group=source.group
  work : d.work=source.work
  configuration : abi.configuration location source d.configuration

/-- The object contract is JOINT: source occurrence, full API arguments,
descriptor/configuration and the decoded dynamic image are one assertion.
Changing an opcode or operand requires a new assertion about the loaded object. -/
abbrev LoadedImage (σ : Type) := SourceLocation → SourceCall σ → CallDescriptor σ → CallImage σ → Prop

/-- The installed registry selects the object/configuration at this actual
ABI boundary AND decoded dynamic image. This is the explicit nonnumerical
loaded-object/decoder correctness contract; it must be witnessed for the
actual resident object, parameters, registers, addresses and states. The
joint relation is defined by this registry and decoder, not independently
assignable for each proof opcode. The registry/decoder are ordinary explicit
compiler and runtime trust boundaries, without task output propositions. -/
structure InstalledLoader (σ : Type) where
  installed : SourceLocation → SourceCall σ → InvocationGraphs.Request σ → CallDescriptor σ → CallImage σ → Prop
  decode : SourceCall σ → InvocationGraphs.Request σ → CallDescriptor σ → CallImage σ

def InstalledLoader.loaded {σ : Type} (loader : InstalledLoader σ) (abi : InvocationABI σ) : LoadedImage σ :=
  fun location source d image => loader.installed location source (abi.request location source) d image ∧
    image=loader.decode source (abi.request location source) d

structure ExactObject {σ : Type} (source : SourceLocation → SourceCall σ → Prop)
    (abi : InvocationABI σ) (loaded : LoadedImage σ) (d : CallDescriptor σ) (image : CallImage σ) where
  location : SourceLocation
  call : SourceCall σ
  member : source location call
  matching : DescriptorMatches abi location call d
  object : loaded location call d image
  states : ImageStates d image

/-- Local loaded-instruction execution contract. Only an active decoded step
of this same loaded image yields an event; it is not an arbitrary registry. -/
structure LoaderContract {σ : Type} (loaded : LoadedImage σ) (trace : PrimitiveTrace σ)
    (permit : Occurrence σ → Opcode σ → Prop) : Prop where
  execute : ∀ location source d image, loaded location source d image →
    ∀ coordinate pc opcode, image.active coordinate pc → image.opcode coordinate pc=some opcode →
      permit ⟨d,coordinate,pc⟩ opcode → Emitted trace ⟨d,coordinate,pc⟩ (image.output coordinate pc) opcode

/-- All selected descriptors are exact source slots and share their image
with the object evidence. The registry is defined below, not supplied. -/
structure ObjectSelection {σ : Type} (source : SourceLocation → SourceCall σ → Prop)
    (abi : InvocationABI σ) (selection : Selection σ) (loaded : LoadedImage σ) where
  image : CallDescriptor σ → CallImage σ
  exact : ∀ d, selection.invoked d → ExactObject source abi loaded d (image d)
  reduction : ∀ site kind width tree, selection.reduction site kind width tree →
    ∃ h : selection.invoked site.call, (exact site.call h).call.kind=reductionCallKind kind

def ObjectSelection.registry {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) : InstructionRegistry σ where
  opcode s := (objects.image s.call).opcode s.coordinate s.instruction
  output s := (objects.image s.call).output s.coordinate s.instruction
  active s := selection.invoked s.call ∧ (objects.image s.call).active s.coordinate s.instruction

theorem ObjectSelection.emitted {σ : Type} {source abi selection loaded trace permit}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (api : LoaderContract loaded trace permit)
    (permission : ∀ s op, objects.registry.active s → objects.registry.opcode s=some op → permit s op)
    (s : Occurrence σ) (op : Opcode σ) (active : objects.registry.active s)
    (opcode : objects.registry.opcode s=some op) : Emitted trace s (objects.registry.output s) op := by
  let actual := objects.exact s.call active.1
  exact api.execute actual.location actual.call s.call (objects.image s.call) actual.object
    s.coordinate s.instruction op active.2 opcode (permission s op active opcode)

def ObjectSelection.instructions {σ : Type} {source abi selection loaded trace permit}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (api : LoaderContract loaded trace permit)
    (permission : ∀ s op, objects.registry.active s → objects.registry.opcode s=some op → permit s op) :
    InstructionExecution objects.registry trace where
  raw s x active opcode := objects.emitted api permission s (.raw x) active opcode
  load s state tensor index active opcode := objects.emitted api permission s (.load state tensor index) active opcode
  literal s x active opcode := objects.emitted api permission s (.literal x) active opcode
  natural s x active opcode := objects.emitted api permission s (.natural x) active opcode
  negativeInfinity s active opcode := objects.emitted api permission s (.negativeInfinity) active opcode
  binary s op a b active opcode := objects.emitted api permission s (.binary op a b) active opcode
  unary s op a active opcode := objects.emitted api permission s (.unary op a) active opcode
  fma s a b c active opcode := objects.emitted api permission s (.fma a b c) active opcode
  hostScale s scale a active opcode := objects.emitted api permission s (.hostScale scale a) active opcode
  transfer s edge a active opcode := objects.emitted api permission s (.transfer edge a) active opcode
  store s value state tensor index active opcode := objects.emitted api permission s (.store value state tensor index) active opcode

/-- Trace completeness in the reverse direction prevents extra, unbound
state-labelled events from being fed directly into return/frame proofs. -/
structure TraceImage {σ : Type} (registry : InstructionRegistry σ) (trace : PrimitiveTrace σ) : Prop where
  event : ∀ s op out, Emitted trace s out op → registry.active s ∧ registry.opcode s=some op ∧
    (op.produces → registry.output s=out)

def ObjectSelection.links {σ : Type} {source abi selection loaded trace}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (events : TraceImage objects.registry trace) :
    TraceCalls selection trace where
  raw s x out event := (events.event s (.raw x) out event).1.1
  load s state tensor index out event := by
    have h := events.event s (.load state tensor index) out event
    exact ⟨h.1.1,((objects.exact s.call h.1.1).states.load s.coordinate s.instruction state tensor index h.1.2 h.2.1).2⟩
  literal s x out event := (events.event s (.literal x) out event).1.1
  natural s x out event := (events.event s (.natural x) out event).1.1
  negativeInfinity s out event := (events.event s (.negativeInfinity) out event).1.1
  binary s op a b out event := (events.event s (.binary op a b) out event).1.1
  unary s op a out event := (events.event s (.unary op a) out event).1.1
  fma s a b c out event := (events.event s (.fma a b c) out event).1.1
  hostScale s scale a out event := (events.event s (.hostScale scale a) out event).1.1
  transfer s edge a out event := (events.event s (.transfer edge a) out event).1.1
  store s value state tensor index event := by
    have h := events.event s (.store value state tensor index) 0 event
    exact ⟨h.1.1,((objects.exact s.call h.1.1).states.store s.coordinate s.instruction value state tensor index h.1.2 h.2.1).2⟩

def ObjectSelection.erasedLoaded {σ : Type} {source abi selection loaded}
    (_objects : ObjectSelection (σ:=σ) source abi selection loaded) : CallKind → String → String → List Nat → Prop :=
  fun kind implementation digest configuration => ∃ location call d image,
    loaded location call d image ∧ call.kind=kind ∧ d.implementation=implementation ∧
      d.objectDigest=digest ∧ d.configuration=configuration

theorem ExactObject.reject_wrong_state {σ : Type} {source abi loaded d image}
    (object : ExactObject (σ:=σ) source abi loaded d image) (state : σ) (wrong : state≠object.call.after.native) :
    state≠d.after := by
  intro same
  exact wrong (same.trans object.matching.after)

theorem InvocationABI.reject_retag {σ : Type} (abi : InvocationABI σ) {a b : SourceLocation} {da db : CallDescriptor σ}
    (ha : da.invocation=abi.identity a) (hb : db.invocation=abi.identity b) (same : da.invocation=db.invocation) : a=b :=
  abi.injective (ha.symm.trans (same.trans hb))

theorem ObjectSelection.reject_opcode {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (s : Occurrence σ) (a b : Opcode σ)
    (decoded : (objects.image s.call).opcode s.coordinate s.instruction=some a) (different : a≠b) :
    objects.registry.opcode s≠some b := by
  intro retag
  exact different (Option.some.inj (decoded.symm.trans retag))

#axiomsClean ObjectSelection.instructions
#axiomsClean ObjectSelection.links
#axiomsClean ExactObject.reject_wrong_state
#axiomsClean ObjectSelection.reject_opcode
#axiomsClean InvocationABI.reject_retag
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
