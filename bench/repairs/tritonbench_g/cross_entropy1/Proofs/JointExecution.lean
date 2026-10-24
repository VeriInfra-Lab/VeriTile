import bench.repairs.tritonbench_g.cross_entropy1.Proofs.InvocationCalls
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.LinkedReturn

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open VeriTile.Triton Source Host Physical
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Identity and routing metadata of one actual transport message. Payload
is one dynamic source-register ID, never a word or numerical result. -/
structure TransportPacket where
  message : Nat
  senderToken : Nat
  receiverToken : Nat
  communicator : Nat
  channel : Nat
  sequence : Nat
  sourceRank : Nat
  destinationRank : Nat
  row : Nat
  payload : Nat
  deriving DecidableEq

/-- One loaded object's dynamic instruction image, for a particular full ABI
and launch. State transitions include setup/cleanup and implementation-local
steps; a kernel's internal native state need not equal the wrapper boundary. -/
structure CallImage (σ : Type) where
  opcode : List Nat → Nat → Option (Opcode σ)
  output : List Nat → Nat → Nat
  active : List Nat → Nat → Prop
  tick : List Nat → Nat → Nat
  logicalIndex : List Nat → Nat → Nat
  cta : List Nat → Nat → Nat
  memoryEpoch : List Nat → Nat → Nat
  length : Nat
  state : Nat → σ
  writes : Nat → List ByteAddress
  /-- Runtime send/receive events at local state boundaries of this same
  decoded dynamic image; these are covered by the joint installed contract. -/
  sent : Nat → TransportPacket → Prop
  received : Nat → TransportPacket → Prop
  requestStarted : Nat → Nat → Prop
  requestFinished : Nat → Nat → Prop

structure ImageStates {σ : Type} (d : CallDescriptor σ) (image : CallImage σ) : Prop where
  initial : image.state 0=d.before
  final : image.state image.length=d.after
  bound : ∀ coordinate pc, image.active coordinate pc → image.tick coordinate pc < image.length
  load : ∀ coordinate pc state tensor index, image.active coordinate pc →
    image.opcode coordinate pc=some (.load state tensor index) →
    state=image.state (image.tick coordinate pc) ∧ tensor∈d.inputs
  store : ∀ coordinate pc value state tensor index, image.active coordinate pc →
    image.opcode coordinate pc=some (.store value state tensor index) →
    state=image.state (image.tick coordinate pc+1) ∧ tensor∈d.outputs

/-- Stream and chosen launch configuration are ordinary runtime metadata for
this full source call. `loaded` below also receives that complete source call. -/
structure InvocationABI (σ : Type) where
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
  installed : SourceLocation → SourceCall σ → CallDescriptor σ → CallImage σ → Prop
  decode : SourceCall σ → CallDescriptor σ → CallImage σ

def InstalledLoader.loaded {σ : Type} (loader : InstalledLoader σ) : LoadedImage σ :=
  fun location source d image => loader.installed location source d image ∧ image=loader.decode source d

structure ExactObject {σ : Type} (source : SourceLocation → SourceCall σ → Prop)
    (abi : InvocationABI σ) (loaded : LoadedImage σ) (d : CallDescriptor σ) (image : CallImage σ) where
  location : SourceLocation
  call : SourceCall σ
  member : source location call
  matching : DescriptorMatches abi location call d
  object : loaded location call d image
  states : ImageStates d image

def Opcode.operands : Opcode σ → List Nat
  | .binary _ a b => [a,b]
  | .unary _ a | .hostScale _ a | .transfer _ a | .store a _ _ _ => [a]
  | .fma a b c => [a,b,c]
  | _ => []

def Opcode.produces : Opcode σ → Prop
  | .store _ _ _ _ => False
  | _ => True

/-- One primitive trace event, retaining opcode, register IDs, address and
state. This is structural and contains no expected arithmetic result. -/
def Emitted (trace : PrimitiveTrace σ) (s : Occurrence σ) (out : Nat) : Opcode σ → Prop
  | .raw x => trace.raw s x out
  | .load state tensor index => trace.load s state tensor index out
  | .literal x => trace.literal s x out
  | .natural x => trace.natural s x out
  | .negativeInfinity => trace.negativeInfinity s out
  | .binary op a b => trace.binary s op a b out
  | .unary op a => trace.unary s op a out
  | .fma a b c => trace.fma s a b c out
  | .hostScale scale a => trace.hostScale s scale a out
  | .transfer edge a => trace.transfer s edge a out
  | .store a state tensor index => trace.store s a state tensor index

/-- Local loaded-instruction execution contract. Only an active decoded step
of this same loaded image yields an event; it is not an arbitrary registry. -/
structure LoaderContract {σ : Type} (loaded : LoadedImage σ) (trace : PrimitiveTrace σ)
    (permit : Occurrence σ → Opcode σ → Prop) : Prop where
  execute : ∀ location source d image, loaded location source d image →
    ∀ coordinate pc opcode, image.active coordinate pc → image.opcode coordinate pc=some opcode →
      permit ⟨d,coordinate,pc⟩ opcode → Emitted trace ⟨d,coordinate,pc⟩ (image.output coordinate pc) opcode

/-- Dynamic register identities and operand ownership. Same-call availability
is ordered by the loaded image's actual instruction ticks. There is no freely
numbered global epoch; cross-call availability is derived from concrete host
and transport paths in CausalExecution/CommunicationProtocol. -/
structure RegisterOrder {σ : Type} (registry : InstructionRegistry σ)
    (image : CallDescriptor σ → CallImage σ) where
  owner : Nat → Occurrence σ
  producer : ∀ s op, registry.active s → registry.opcode s=some op → op.produces →
    owner (registry.output s)=s
  operands : ∀ s op, registry.active s → registry.opcode s=some op → ∀ value∈op.operands,
    registry.active (owner value) ∧ registry.output (owner value)=value
  localOperands : ∀ s op, registry.active s → registry.opcode s=some op →
    (∀ edge value, op≠Opcode.transfer edge value) → ∀ value∈op.operands, (owner value).call=s.call
  earlier : ∀ s op, registry.active s → registry.opcode s=some op → ∀ value∈op.operands,
    (owner value).call=s.call →
    (image (owner value).call).tick (owner value).coordinate (owner value).instruction <
      (image s.call).tick s.coordinate s.instruction

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

def ObjectSelection.calls {σ : Type} {source abi selection loaded} {allowed : Runtime Expr σ → Prop}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded)
    (permitted : ∀ location call, source location call → allowed call.runtime) :
    PlatformCalls selection allowed objects.erasedLoaded where
  invocation d h := by
    let actual := objects.exact d h
    refine ⟨actual.call.runtime,permitted _ _ actual.member,actual.call.kind,actual.call.before,actual.call.after,
      actual.matching.before,actual.matching.after,?_,?_⟩
    · rw [actual.matching.inputs,actual.matching.outputs,actual.matching.group,actual.matching.work]
      exact actual.call.actual
    · exact ⟨actual.location,actual.call,d,objects.image d,actual.object,rfl,rfl,rfl,rfl⟩
  reduction site kind width tree selected := objects.reduction site kind width tree selected

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
#axiomsClean ObjectSelection.calls
#axiomsClean ExactObject.reject_wrong_state
#axiomsClean ObjectSelection.reject_opcode
#axiomsClean InvocationABI.reject_retag
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
