import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Launches
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Scalar
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Contract
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Physical (Role)
noncomputable section
open Classical

structure World (M : Type) where
  memory : M
  currentDevice : Nat
  currentStreams : Nat → Nat
structure Queue where
  owner : Nat
  stream : Nat
structure Context where
  previous : Nat

def World.queue {M : Type} (s : World M) (owner : Nat) : Queue := ⟨owner,s.currentStreams owner⟩

inductive Result (M A : Type) where
  | error (reason : String) (state : World M)
  | ok (value : A) (state : World M)
def Result.state {M A : Type} : Result M A → World M
  | .error _ s | .ok _ s => s
def Result.bind {M A B : Type} (first : Result M A) (next : A → World M → Result M B) : Result M B :=
  match first with | .error why s => .error why s | .ok a s => next a s
def Ensures {M A : Type} (ok : A → World M → Prop) : Result M A → Prop
  | .error _ _ => True | .ok value after => ok value after

structure Request where
  p : View
  grad : View
  momentum : View

def Request.count (q : Request) : Nat := q.p.numel
def Request.caller (q : Request) : List Allocation := [q.p.storage,q.grad.storage,q.momentum.storage]
def Request.view (q : Request) : Role → View
  | .a => q.p | .b => q.momentum | .third => q.grad

def outputView (q : Request) : Output → View
  | .parameter => q.p | .momentum => q.momentum

def CallerBytes (q : Request) (owner address : Nat) : Prop :=
  ∃ old∈q.caller,old.covers owner address

def Outside (q : Request) (owner address : Nat) : Prop :=
  ∀ output i,i<q.count → ¬((outputView q output).key i).covers owner address

structure Buffers where
  parameter : Allocation
  gradient : Allocation
  momentum : Allocation

def Buffers.layout (b : Buffers) (q : Request) : Private.Layout :=
  ⟨(fun | .a => b.parameter | .b => b.momentum | .third => b.gradient),q.count⟩

/-- A logical contiguous window of the original allocation; no fresh storage
or numerical value callback is involved. Singleton strides need not be one. -/
structure Window (input : View) (a : Allocation) : Prop where
  storageObject : a.object=input.storage.object
  owner : a.owner=input.storage.owner
  bytes : a.bytes=4*input.numel
  keys : ∀ i,i < input.numel → a.key i=input.key i

structure Runtime (M : Type) where
  live : World M → Allocation → Prop
  enter : Nat → World M → Result M Context
  exit : Context → World M → Result M Unit
  flatten : View → World M → Result M Allocation

structure Library {M : Type} (rt : Runtime M) : Prop where
  enter : ∀ owner before token after,rt.enter owner before=.ok token after →
    token.previous=before.currentDevice ∧ after.memory=before.memory ∧
    after.currentDevice=owner ∧ after.currentStreams=before.currentStreams ∧
    ∀ old,rt.live before old → rt.live after old
  exit : ∀ token before after,rt.exit token before=.ok () after →
    after.memory=before.memory ∧ after.currentDevice=token.previous ∧
    after.currentStreams=before.currentStreams ∧ ∀ old,rt.live before old → rt.live after old
  flatten : ∀ input before out after,rt.flatten input before=.ok out after →
    Window input out ∧ after=before

def prepare {M : Type} (rt : Runtime M) (q : Request) (s : World M) : Result M Buffers :=
  (rt.flatten q.p s).bind fun p s1 =>
  (rt.flatten q.grad s1).bind fun g s2 =>
  (rt.flatten q.momentum s2).bind fun m s3 => .ok ⟨p,g,m⟩ s3

def finish {M A : Type} (rt : Runtime M) (token : Context) : Result M A → Result M A
  | .error reason s => (rt.exit token s).bind fun _ after => .error reason after
  | .ok value s => (rt.exit token s).bind fun _ after => .ok value after

def contextual {M A : Type} (rt : Runtime M) (owner : Nat)
    (body : World M → Result M A) (s : World M) : Result M A :=
  (rt.enter owner s).bind fun token entered => finish rt token (body entered)

structure ContextPath {M A : Type} (rt : Runtime M) (owner : Nat)
    (body : World M → Result M A) (before : World M) (value : A) (after : World M) where
  token : Context
  entered : World M
  completed : World M
  enter : rt.enter owner before=.ok token entered
  body : body entered=.ok value completed
  exit : rt.exit token completed=.ok () after

theorem contextual_path {M A : Type} (rt : Runtime M) (owner : Nat)
    (body : World M → Result M A) (s : World M) :
    Ensures (fun value after => Nonempty (ContextPath rt owner body s value after)) (contextual rt owner body s) := by
  unfold contextual
  cases entered : rt.enter owner s with
  | error => trivial
  | ok token inside =>
    simp only [Result.bind]
    cases done : body inside with
    | error reason failed =>
      simp only [finish]
      cases rt.exit token failed <;> trivial
    | ok value completed =>
      simp only [finish]
      cases exited : rt.exit token completed with
      | error => trivial
      | ok unit after => cases unit; exact ⟨⟨token,inside,completed,entered,done,exited⟩⟩

/-- Failed bodies keep their actual partial memory when context exit succeeds. -/
theorem finish_failure {M A : Type} (rt : Runtime M) (token : Context) (reason : String)
    (failed restored : World M) (exited : rt.exit token failed=.ok () restored) :
    finish (A:=A) rt token (.error reason failed)=.error reason restored := by
  simp [finish,exited,Result.bind]

/-- A single actual autotuned source launch, including all successful trials,
restores, and the selected source execution. This is an execution interface,
not a numerical assumption or a callback for the Lion answer. -/
structure ChunkRuntime (M α τ : Type) [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Scalars α → Machine.Model α τ) (rt : Runtime M) where
  run : Scalars α → Request → Buffers → Index.Chunk → World M → Result M Unit
  source : ∀ params q b chunk before after,run params q b chunk before=.ok () after →
    Nonempty (Autotune.Launched ops (model params) (b.layout q) chunk (CallerBytes q)
      before.memory after.memory)
  live : ∀ params q b chunk before after,run params q b chunk before=.ok () after →
    ∀ old,rt.live before old → rt.live after old
  device : ∀ params q b chunk before after,run params q b chunk before=.ok () after →
    after.currentDevice=before.currentDevice
  streams : ∀ params q b chunk before after,run params q b chunk before=.ok () after →
    after.currentStreams=before.currentStreams

def scalarFailure (failure : Scalar.Failure) : String :=
  (match failure.argument with | .lr => "lr" | .wd => "wd" | .beta1 => "beta1" | .beta2 => "beta2") ++
    ": " ++ failure.reason

/-- Binder/setup is encountered separately for every actual chunk, after the three
direct input views. A failure carries the current state, including any earlier
in-place launches; no rollback is assumed. -/
def launchChunks {M α β τ : Type} [Inhabited α] [Inhabited τ]
    {ops : Ops M α β} {model : Scalars α → Machine.Model α τ} {rt : Runtime M}
    (driver : ChunkRuntime M α τ ops model rt) (convert : Scalar.Conversions α)
    (args : Scalars (BitVec 64)) (q : Request) (b : Buffers) :
    List Index.Chunk → World M → Result M Unit
  | [],s => .ok () s
  | chunk::rest,s =>
    match Scalar.bind convert args with
    | .error reason => .error (scalarFailure reason) s
    | .ok params => (driver.run params q b chunk s).bind fun _ next =>
        launchChunks driver convert args q b rest next

structure ChunksFacts {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Machine.Model α τ) (rt : Runtime M)
    (q : Request) (b : Buffers) (parts : List Index.Chunk) (before after : World M) : Prop where
  source : Autotune.Launches ops model (b.layout q) (CallerBytes q) parts before.memory after.memory
  live : ∀ old,rt.live before old → rt.live after old
  device : after.currentDevice=before.currentDevice
  streams : after.currentStreams=before.currentStreams

theorem chunks_facts {M α β τ : Type} [Inhabited α] [Inhabited τ]
    {ops : Ops M α β} {model : Scalars α → Machine.Model α τ} {rt : Runtime M}
    (driver : ChunkRuntime M α τ ops model rt) (convert : Scalar.Conversions α)
    (args : Scalars (BitVec 64)) (params : Scalars α) (binding : Scalar.bind convert args=.ok params)
    (q : Request) (b : Buffers) (parts : List Index.Chunk) (before after : World M)
    (ran : launchChunks driver convert args q b parts before=.ok () after) :
    ChunksFacts ops (model params) rt q b parts before after := by
  induction parts generalizing before with
  | nil =>
    simp only [launchChunks,Result.ok.injEq] at ran
    rcases ran with ⟨_,rfl⟩
    exact ⟨.nil _,fun _ h => h,rfl,rfl⟩
  | cons chunk rest ih =>
    simp only [launchChunks,binding] at ran
    cases step : driver.run params q b chunk before with
    | error reason failed => simp [step,Result.bind] at ran
    | ok unit middle =>
      cases unit
      simp only [step,Result.bind] at ran
      have tail := ih middle ran
      obtain ⟨head⟩ := driver.source params q b chunk before middle step
      exact ⟨.cons chunk rest _ _ _ head tail.source,
        fun old h => tail.live old (driver.live params q b chunk before middle step old h),
        tail.device.trans (driver.device params q b chunk before middle step),
        tail.streams.trans (driver.streams params q b chunk before middle step)⟩

def pipeline {M α β τ : Type} [Inhabited α] [Inhabited τ]
    {ops : Ops M α β} {model : Scalars α → Machine.Model α τ} (rt : Runtime M)
    (driver : ChunkRuntime M α τ ops model rt) (convert : Scalar.Conversions α)
    (args : Scalars (BitVec 64)) (q : Request) (s : World M) : Result M Unit :=
  (prepare rt q s).bind fun b prepared =>
  launchChunks driver convert args q b (Index.chunks 0 q.count) prepared

structure Call where
  parameter : TensorMetadata
  gradient : TensorMetadata
  momentum : TensorMetadata
  request : Request
  arguments : Scalars Scalar.Argument

def Admitted (call : Call) : Prop := InputContract call.parameter call.gradient call.momentum

structure Bindings (call : Call) : Prop where
  parameter : TensorBinding call.parameter call.request.p
  gradient : TensorBinding call.gradient call.request.grad
  momentum : TensorBinding call.momentum call.request.momentum

def run {M α β τ : Type} [Inhabited α] [Inhabited τ]
    {ops : Ops M α β} {model : Scalars α → Machine.Model α τ} (rt : Runtime M)
    (driver : ChunkRuntime M α τ ops model rt) (convert : Scalar.Conversions α)
    (call : Call) (s : World M) : Result M Unit :=
  match checkInputs (.tensor call.parameter) (.tensor call.gradient) (.tensor call.momentum) with
  | .error _ => .error "unsupported tensor inputs" s
  | .ok _ =>
    if call.request.count=0 then .ok () s
    else match Scalar.normalize convert call.arguments with
    | .error why => .error (scalarFailure why) s
    | .ok args => contextual rt call.request.p.storage.owner (pipeline rt driver convert args call.request) s

#axiomsClean chunks_facts
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Host
