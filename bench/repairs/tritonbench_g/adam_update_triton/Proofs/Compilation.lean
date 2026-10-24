import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Preparation
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Compilation
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune (Config)
open Host
noncomputable section

def candidateHash : String := "0a9e638a151a19cacce590512938f65c64a753d67d22a29cd88e601ccca4404f"

inductive CountABI where
  | constexprOne | signed32 | signed64
  deriving DecidableEq

def countABI (count : Nat) : CountABI :=
  if count=1 then .constexprOne else if count<2147483648 then .signed32 else .signed64

/-- Full resolved options are retained verbatim in the profile, in addition to
fields relevant to this graph. Evidence pins include signatures/constants,
attributes, source, TTIR/TTGIR/LLVM IR/PTX/cubin and their exact hashes. -/
structure Profile where
  sourceSHA256 : String
  backend : String
  architecture : Nat
  compiler : String
  fpFusion : Bool
  reflectFTZ : Bool
  fullOptions : String
  optionsSHA256 : String
  scalarABI : List String
  totalABI : CountABI
  chunkABI : CountABI
  block : Nat
  warps : Nat
  evidenceSHA256 : String

inductive Lowering where
  | observedLionGraph
  | different (description : String)
  deriving DecidableEq

structure Resolved where
  profile : Profile
  instructions : Native.Instructions Native.Word
  lowering : Lowering

/-- Every trial/selected invocation identifies its exact original key, local
mask count, configuration, current queue, pointer slice and bound FP32 words. -/
structure Invocation where
  scalars : Scalars Native.Word
  total : Nat
  chunk : Index.Chunk
  config : Config
  queue : Queue
  buffers : Buffers

structure Compiler (M : Type) where
  resolve : Invocation → M → Resolved
  profile : Invocation → Profile
  architecture : Nat → Nat
  version : String
  fpFusion : Bool
  reflectFTZ : Bool

def invocation (params : Scalars Native.Word) (q : Request) (b : Buffers)
    (chunk : Index.Chunk) (config : Config) (queue : Queue) : Invocation :=
  ⟨params,q.count,chunk,config,queue,b⟩

def warps (config : Config) : Nat := if config.block=128 then 4 else 8

/-- A setup-local compiler/cache boundary, never a final-value or memory law.
Samples pin CUDA/sm90/Triton3.7.1 only. Other resolutions must supply this exact
local graph contract; offline samples do not establish universal conformance. -/
structure Binding {M : Type} (compiler : Compiler M) (native : Native.Instructions Native.Word)
    (inv : Invocation) (state : M) : Prop where
  source : (compiler.profile inv).sourceSHA256=candidateHash
  profile : (compiler.resolve inv state).profile=compiler.profile inv
  instructions : (compiler.resolve inv state).instructions=native
  lowering : (compiler.resolve inv state).lowering=.observedLionGraph
  backend : (compiler.profile inv).backend="cuda"
  architecture : (compiler.profile inv).architecture=compiler.architecture inv.queue.owner
  version : (compiler.profile inv).compiler=compiler.version
  fusion : (compiler.profile inv).fpFusion=compiler.fpFusion
  reflectFTZ : (compiler.profile inv).reflectFTZ=compiler.reflectFTZ
  scalarABI : (compiler.profile inv).scalarABI=["fp32","fp32","fp32","fp32"]
  totalABI : (compiler.profile inv).totalABI=countABI inv.total
  chunkABI : (compiler.profile inv).chunkABI=countABI inv.chunk.count
  block : (compiler.profile inv).block=inv.config.block
  warps : (compiler.profile inv).warps=warps inv.config

/-- Resolve the executable selected at each source invocation. This relation
is intentionally independent of any numerical target or final memory result. -/
structure Local {M : Type} (compiler : Compiler M) (native : Native.Instructions Native.Word)
    (params : Scalars Native.Word) (q : Request) (b : Buffers) (queue : Queue) : Prop where
  resolve : ∀ chunk config state,Binding compiler native (invocation params q b chunk config queue) state

inductive TrialsBound {M β : Type} (ops : Ops M Native.Word β) (compiler : Compiler M)
    (native : Native.Instructions Native.Word) (params : Scalars Native.Word)
    (q : Request) (b : Buffers) (queue : Queue) (chunk : Index.Chunk) :
    {history : List Config} → {before after : M} →
      Autotune.Trials ops (Machine.nativeModel native params) (b.layout q) chunk (CallerBytes q)
        history before after → Prop
  | nil (state) : TrialsBound ops compiler native params q b queue chunk (.nil state)
  | cons {config rest schedule before prepared trial restored after} {prepare} {source} {restore} {tail}
      (head : Binding compiler native (invocation params q b chunk config queue) prepared)
      (next : TrialsBound ops compiler native params q b queue chunk tail) :
      TrialsBound ops compiler native params q b queue chunk
        (.cons config rest schedule before prepared trial restored after prepare source restore tail)
  | administrative {history before prepared after} {effect} {tail}
      (next : TrialsBound ops compiler native params q b queue chunk tail) :
      TrialsBound ops compiler native params q b queue chunk
        (.administrative history before prepared after effect tail)

def LaunchBound {M β : Type} (ops : Ops M Native.Word β) (compiler : Compiler M)
    (native : Native.Instructions Native.Word) (params : Scalars Native.Word)
    (q : Request) (b : Buffers) (queue : Queue) (chunk : Index.Chunk) (before after : M)
    (launch : Autotune.Launched ops (Machine.nativeModel native params) (b.layout q) chunk (CallerBytes q) before after) : Prop :=
  TrialsBound ops compiler native params q b queue chunk launch.trials ∧
  Binding compiler native (invocation params q b chunk launch.selected queue) launch.restored

theorem trials_bound {M β : Type} (ops : Ops M Native.Word β) (compiler : Compiler M)
    (native : Native.Instructions Native.Word) (params : Scalars Native.Word)
    (q : Request) (b : Buffers) (queue : Queue) (contracts : Local compiler native params q b queue)
    (chunk : Index.Chunk) (history : List Config) (before after : M)
    (trials : Autotune.Trials ops (Machine.nativeModel native params) (b.layout q) chunk (CallerBytes q) history before after) :
    TrialsBound ops compiler native params q b queue chunk trials := by
  induction trials with
  | nil state => exact .nil state
  | cons config rest schedule before prepared trial restored after prepare source restore tail ih =>
    exact .cons (prepare:=prepare) (source:=source) (restore:=restore)
      (contracts.resolve chunk config prepared) ih
  | administrative history before prepared after effect tail ih =>
    exact .administrative (effect:=effect) ih

theorem launch_bound {M β : Type} (ops : Ops M Native.Word β) (compiler : Compiler M)
    (native : Native.Instructions Native.Word) (params : Scalars Native.Word)
    (q : Request) (b : Buffers) (queue : Queue) (contracts : Local compiler native params q b queue)
    (chunk : Index.Chunk) (before after : M)
    (launch : Autotune.Launched ops (Machine.nativeModel native params) (b.layout q) chunk (CallerBytes q) before after) :
    LaunchBound ops compiler native params q b queue chunk before after launch :=
  ⟨trials_bound ops compiler native params q b queue contracts chunk _ _ _ launch.trials,
    contracts.resolve chunk launch.selected launch.restored⟩

inductive LaunchesBound {M β : Type} (ops : Ops M Native.Word β) (compiler : Compiler M)
    (native : Native.Instructions Native.Word) (params : Scalars Native.Word)
    (q : Request) (b : Buffers) (queue : Queue) :
    {parts : List Index.Chunk} → {before after : M} →
      Autotune.Launches ops (Machine.nativeModel native params) (b.layout q) (CallerBytes q)
        parts before after → Prop
  | nil (state) : LaunchesBound ops compiler native params q b queue (.nil state)
  | cons {chunk rest before middle after} {head} {tail}
      (first : LaunchBound ops compiler native params q b queue chunk before middle head)
      (next : LaunchesBound ops compiler native params q b queue tail) :
      LaunchesBound ops compiler native params q b queue (.cons chunk rest before middle after head tail)

theorem launches_bound {M β : Type} (ops : Ops M Native.Word β) (compiler : Compiler M)
    (native : Native.Instructions Native.Word) (params : Scalars Native.Word)
    (q : Request) (b : Buffers) (queue : Queue) (contracts : Local compiler native params q b queue)
    (parts : List Index.Chunk) (before after : M)
    (launches : Autotune.Launches ops (Machine.nativeModel native params) (b.layout q) (CallerBytes q) parts before after) :
    LaunchesBound ops compiler native params q b queue launches := by
  induction launches with
  | nil state => exact .nil state
  | cons chunk rest before middle after head tail ih =>
    exact .cons (launch_bound ops compiler native params q b queue contracts chunk before middle head) ih

/-- The host C launcher multiplies (gridX,1,1), so its signed-int product fits.
This bound is derived from the existing chunk rule, with no new input cap. -/
theorem launcher_product (count : Nat) (config : Config) (bounded : count≤8388480) :
    Index.grid count config*1*1<2147483648 := by
  have := Index.grid_bound count config bounded
  omega

#axiomsClean trials_bound
#axiomsClean launch_bound
#axiomsClean launches_bound
#axiomsClean launcher_product
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Compilation
