import bench.repairs.tritonbench_g.spinning_lock_reduction.Controller

/-! Task-local typed source subset for the two spinning-kernel specializations.
The public Triton DSL has no while semantics. This reader therefore keeps an
explicit SC, row-major lane model, reusing the historical memory executor.
Its compiler extracts expressions from every source statement; the executable
Plan reads those expressions, rather than selecting a handwritten controller.
Atomics expose their actual returned old word. The barrier is an explicit
whole-block boundary after all producer lane stores. CUDA/compiler refinement
and GPU scheduling remain external model boundaries. -/
namespace VeriTile.Bench.Repairs.SpinningLock.SourceReader
open VeriTile.Bench.TritonBenchG.SpinningLock

inductive Param where
  | numSms | groupK | m | n | strideM | strideN | bm | bn | phase
  deriving DecidableEq, BEq, Repr

inductive IntExpr where
  | literal (n : Nat)
  | param (p : Param)
  | variable (name : String)
  | pid | next | row | col
  | signed (x : IntExpr) | unsigned (x : IntExpr)
  | add (a b : IntExpr) | mul (a b : IntExpr)
  | div (a b : IntExpr) | rem (a b : IntExpr)
  deriving DecidableEq, BEq, Repr

def IntExpr.isUnsigned : IntExpr → Bool
  | .unsigned _ => true
  | .add a b | .mul a b | .div a b | .rem a b => a.isUnsigned || b.isUnsigned
  | _ => false

inductive BoolExpr where
  | variable (name : String)
  | eq (a b : IntExpr) | ne (a b : IntExpr) | lt (a b : IntExpr)
  | and (a b : BoolExpr) | or (a b : BoolExpr) | not (a : BoolExpr)
  deriving DecidableEq, BEq, Repr

inductive Region where | partials | output | locks
  deriving DecidableEq, BEq, Repr

inductive Pointer where
  | base (r : Region)
  | variable (name : String)
  | offset (p : Pointer) (x : IntExpr)
  deriving DecidableEq, BEq, Repr

inductive Condition where
  | pure (test : BoolExpr)
  | casNe (pointer : Pointer) (compare replace expected : IntExpr)
  deriving DecidableEq, BEq, Repr

inductive Statement where
  | integer (name : String) (value : IntExpr)
  | boolean (name : String) (value : BoolExpr)
  | pointer (name : String) (value : Pointer)
  | staticAssert (test : BoolExpr)
  | zeroFP32 (name : String) (rows columns : Param)
  | loadFP32 (name : String) (pointer : Pointer)
  | addFP32 (target input : String)
  | storeFP32 (pointer : Pointer) (value : String) (mask : Option BoolExpr)
  | xchg (pointer : Pointer) (value : IntExpr)
  | blockBarrier
  | forRange (index : String) (start stop : Nat) (body : List Statement)
  | branch (test : BoolExpr) (yes no : List Statement)
  | whileLoop (test : Condition) (body : List Statement)
  | pass | returnNow
  deriving Repr

abbrev IntEnv := String → Option IntExpr
abbrev BoolEnv := String → Option BoolExpr
abbrev PtrEnv := String → Option Pointer

def resolveI (env : IntEnv) : IntExpr → Option IntExpr
  | .variable n => env n
  | .signed x => return .signed (← resolveI env x)
  | .unsigned x => return .unsigned (← resolveI env x)
  | .add a b => return .add (← resolveI env a) (← resolveI env b)
  | .mul a b => return .mul (← resolveI env a) (← resolveI env b)
  | .div a b => return .div (← resolveI env a) (← resolveI env b)
  | .rem a b => return .rem (← resolveI env a) (← resolveI env b)
  | e => some e

def resolveB (ienv : IntEnv) (benv : BoolEnv) : BoolExpr → Option BoolExpr
  | .variable n => benv n
  | .eq a b => return .eq (← resolveI ienv a) (← resolveI ienv b)
  | .ne a b => return .ne (← resolveI ienv a) (← resolveI ienv b)
  | .lt a b => return .lt (← resolveI ienv a) (← resolveI ienv b)
  | .and a b => return .and (← resolveB ienv benv a) (← resolveB ienv benv b)
  | .or a b => return .or (← resolveB ienv benv a) (← resolveB ienv benv b)
  | .not a => return .not (← resolveB ienv benv a)

def resolveP (ienv : IntEnv) (penv : PtrEnv) : Pointer → Option Pointer
  | .base r => some (.base r)
  | .variable n => penv n
  | .offset p e => return .offset (← resolveP ienv penv p) (← resolveI ienv e)

/-- The schema fixes only supported instruction forms and local names. Every
executed bound/condition/address comes from an extracted source expression. -/
structure Plan where
  assertion : BoolExpr
  skip : BoolExpr
  start : Nat
  stop : Nat
  reducer : BoolExpr
  nextInitial : IntExpr
  advance : IntExpr
  continues : BoolExpr
  poll : Pointer
  compare : IntExpr
  replace : IntExpr
  expected : IntExpr
  load : Pointer
  publish : Pointer
  release : Pointer
  releaseWord : IntExpr
  output : Pointer
  mask : BoolExpr
  deriving DecidableEq, BEq, Repr

def regionOf : Pointer → Option Region
  | .base r => some r
  | .offset p _ => regionOf p
  | .variable _ => none

/-- Compilation fails for unsupported source forms. It checks the entire tree,
including both while bodies, barrier, store payloads and initialization; it
does not take successful execution or any result equation as an argument. -/
def compile : List Statement → Option Plan
  | [.integer "pid" pid, .staticAssert assertion,
     .boolean "is_reducer" role, .branch skip [.returnNow] [],
     .integer "pid_m" pidm, .integer "pid_n" pidn,
     .zeroFP32 "acc" .bm .bn, .integer "lock_ready" ready,
     .forRange "iters" start stop
       [.branch branch
         [.integer "next_pid" nextInitial,
          .whileLoop (.pure continues)
            [.whileLoop (.casNe poll compare replace expected) [.pass],
             .integer "rm1" lr, .integer "rn1" lc,
             .pointer "P_" lp, .loadFP32 "acc1" lp',
             .addFP32 "acc" "acc1", .integer "next_pid" advance]]
         [.integer "rm1" pr, .integer "rn1" pc,
          .pointer "P_" pp, .storeFP32 pp' "acc" none,
          .blockBarrier, .xchg release releaseWord],
        .integer "rm" outrow, .integer "rn" outcol,
        .pointer "C_" output, .boolean "mask" mask,
        .storeFP32 output' "acc" (some mask')]] => do
      let emptyI : IntEnv := fun _ => none
      let emptyB : BoolEnv := fun _ => none
      let emptyP : PtrEnv := fun _ => none
      let pid ← resolveI emptyI pid
      let i₀ := put emptyI "pid" (some pid)
      let assertion ← resolveB i₀ emptyB assertion
      let role ← resolveB i₀ emptyB role
      let b₀ := put emptyB "is_reducer" (some role)
      let skip ← resolveB i₀ b₀ skip
      let pidm ← resolveI i₀ pidm
      let pidn ← resolveI i₀ pidn
      let ready ← resolveI i₀ ready
      let i₁ := put (put (put i₀ "pid_m" (some pidm)) "pid_n" (some pidn)) "lock_ready" (some ready)
      let branch ← resolveB i₁ b₀ branch
      let nextInitial ← resolveI i₁ nextInitial
      let i₂ := put i₁ "next_pid" (some .next)
      let continues ← resolveB i₂ b₀ continues
      let poll ← resolveP i₂ emptyP poll
      let compare ← resolveI i₂ compare
      let replace ← resolveI i₂ replace
      let expected ← resolveI i₂ expected
      let lr ← resolveI i₂ lr
      let lc ← resolveI i₂ lc
      let li := put (put i₂ "rm1" (some lr)) "rn1" (some lc)
      let load ← resolveP li emptyP lp
      let load' ← resolveP li (put emptyP "P_" (some load)) lp'
      let advance ← resolveI i₂ advance
      let pr ← resolveI i₁ pr
      let pc ← resolveI i₁ pc
      let pi := put (put i₁ "rm1" (some pr)) "rn1" (some pc)
      let publish ← resolveP pi emptyP pp
      let publish' ← resolveP pi (put emptyP "P_" (some publish)) pp'
      let release ← resolveP i₁ emptyP release
      let releaseWord ← resolveI i₁ releaseWord
      let outrow ← resolveI i₁ outrow
      let outcol ← resolveI i₁ outcol
      let oi := put (put i₁ "rm" (some outrow)) "rn" (some outcol)
      let output ← resolveP oi emptyP output
      let output' ← resolveP oi (put emptyP "C_" (some output)) output'
      let mask ← resolveB oi b₀ mask
      let mask' ← resolveB oi (put b₀ "mask" (some mask)) mask'
      if load==load' && publish==publish' && output==output' && mask==mask' &&
          regionOf poll==some .locks && regionOf release==some .locks &&
          regionOf load==some .partials && regionOf publish==some .partials && regionOf output==some .output
      then some ⟨assertion,skip,start,stop,branch,nextInitial,advance,continues,
        poll,compare,replace,expected,load,publish,release,releaseWord,output,mask⟩
      else none
  | _ => none

def phaseNumber : Phased.Stage → Nat | .producers => 1 | .consumers => 2

def evalI (h : HostArgs) (stage : Phased.Stage) (pid next lane : Nat) : IntExpr → BitVec 64
  | .literal n => BitVec.ofNat 64 n
  | .param p => BitVec.ofNat 64 (match p with
      | .numSms => h.numSms | .groupK => h.groupK | .m => h.m | .n => h.n
      | .strideM => h.strideM | .strideN => h.strideN | .bm => h.bm | .bn => h.bn
      | .phase => phaseNumber stage)
  | .variable _ => 0 -- Unresolved variables are rejected by compile.
  | .pid => BitVec.ofNat 64 pid
  | .next => BitVec.ofNat 64 next
  | .row => BitVec.ofNat 64 (lane/h.bn)
  | .col => BitVec.ofNat 64 (lane%h.bn)
  | .signed x | .unsigned x => evalI h stage pid next lane x
  | .add a b => evalI h stage pid next lane a + evalI h stage pid next lane b
  | .mul a b => evalI h stage pid next lane a * evalI h stage pid next lane b
  | .div a b => if a.isUnsigned || b.isUnsigned then
      (evalI h stage pid next lane a).udiv (evalI h stage pid next lane b)
      else (evalI h stage pid next lane a).sdiv (evalI h stage pid next lane b)
  | .rem a b => if a.isUnsigned || b.isUnsigned then
      (evalI h stage pid next lane a).umod (evalI h stage pid next lane b)
      else (evalI h stage pid next lane a).srem (evalI h stage pid next lane b)

def evalB (h : HostArgs) (stage : Phased.Stage) (pid next lane : Nat) : BoolExpr → Bool
  | .variable _ => false
  | .eq a b => evalI h stage pid next lane a == evalI h stage pid next lane b
  | .ne a b => evalI h stage pid next lane a != evalI h stage pid next lane b
  | .lt a b => if a.isUnsigned || b.isUnsigned then
      (evalI h stage pid next lane a).ult (evalI h stage pid next lane b)
      else (evalI h stage pid next lane a).slt (evalI h stage pid next lane b)
  | .and a b => evalB h stage pid next lane a && evalB h stage pid next lane b
  | .or a b => evalB h stage pid next lane a || evalB h stage pid next lane b
  | .not a => !evalB h stage pid next lane a

/-- Each pointer addition sign-extends its own signed64 element offset. -/
def address (h : HostArgs) (stage : Phased.Stage) (pid next lane : Nat) : Pointer → Int
  | .base _ | .variable _ => 0
  | .offset p x => address h stage pid next lane p + (evalI h stage pid next lane x).toInt

def word (x : BitVec 64) : I32 := BitVec.ofNat 32 x.toNat
def cas (old compare replace : I32) : I32 × I32 :=
  (old,if old==compare then replace else old)

inductive SourceEvent where
  | memory (event : Event)
  | cas (address : Int) (compare replace : I32)
  | exchange (address : Int) (value : I32)
  | barrier
  deriving DecidableEq, Repr

def project : SourceEvent → Event
  | .memory e => e | .cas a _ _ => .poll a | .exchange a _ => .release a | .barrier => .silent

def executeSource (b : Backend V) (e : SourceEvent) (s : State V) : State V :=
  match e with
  | .memory e => execute b e s
  | .cas a cmp repl => {s with mem.locks := put s.mem.locks a (cas (s.mem.locks a) cmp repl).2}
  | .exchange a value => {s with mem.locks := put s.mem.locks a value}
  | .barrier => s

def decodePlan (plan : Plan) (h : HostArgs) (stage : Phased.Stage)
    (p : Nat) (s : Wide.State V) : SourceEvent × Wide.Control :=
  let ctl := s.pc p
  if p≥h.numSms then (.memory .silent,ctl) else
  match ctl.phase with
  | .branch =>
    if ctl.round≥plan.stop then (.memory .silent,{ctl with phase := .done}) else
    let nextPhase : Wide.Phase := if evalB h stage p 0 0 plan.reducer then
      .scan ((evalI h stage p 0 0 plan.nextInitial).toNat) else .publish 0
    (.memory .silent, {ctl with phase := nextPhase})
  | .scan next =>
    let nextPhase : Wide.Phase := if evalB h stage p next 0 plan.continues then .poll next else .output 0
    (.memory .silent, {ctl with phase := nextPhase})
  | .poll next =>
    let a := address h stage p next 0 plan.poll
    let cmp := word (evalI h stage p next 0 plan.compare)
    let repl := word (evalI h stage p next 0 plan.replace)
    let expected := word (evalI h stage p next 0 plan.expected)
    let returnedOld := (cas (s.values.mem.locks a) cmp repl).1
    (.cas a cmp repl,{ctl with phase := if returnedOld==expected then .gather next 0 else .poll next})
  | .gather next lane =>
    if lane<Wide.lanes h then
      (.memory (.add p lane (address h stage p next lane plan.load)),{ctl with phase := .gather next (lane+1)})
    else (.memory .silent,{ctl with phase := .scan (evalI h stage p next lane plan.advance).toNat})
  | .publish lane =>
    if lane<Wide.lanes h then
      (.memory (.publish p lane (address h stage p 0 lane plan.publish)),{ctl with phase := .publish (lane+1)})
    else (.barrier,{ctl with phase := .release})
  | .release => (.exchange (address h stage p 0 0 plan.release)
      (word (evalI h stage p 0 0 plan.releaseWord)),{ctl with phase := .output 0})
  | .output lane =>
    if lane<Wide.lanes h then
      (if evalB h stage p 0 lane plan.mask then
          .memory (.output p lane (address h stage p 0 lane plan.output)) else .memory .silent,
        {ctl with phase := .output (lane+1)})
    else (.memory .silent,{round := ctl.round+1,phase := .branch})
  | .done | .fault => (.memory .silent,ctl)

def stepPlan (b : Backend V) (plan : Plan) (h : HostArgs) (stage : Phased.Stage)
    (p : Nat) (s : Wide.State V) : SourceEvent × Wide.State V :=
  let (event,ctl) := decodePlan plan h stage p s
  (event,⟨executeSource b event s.values,put s.pc p ctl⟩)

def initialPlan (b : Backend V) (plan : Plan) (mem : Memory V) : Wide.State V :=
  ⟨initial b mem,fun _ => ⟨plan.start,.branch⟩⟩

/-- The early-return guard has no memory instruction. Skipped local control
slots are ghost state. Every selected step executes the extracted source plan. -/
def runPlan (b : Backend V) (plan : Plan) (h : HostArgs) (stage : Phased.Stage) :
    List Nat → Wide.State V → List SourceEvent × Wide.State V
  | [],s => ([],s)
  | p::ps,s =>
      if p≥h.numSms || evalB h stage p 0 0 plan.skip then runPlan b plan h stage ps s else
      let (event,next) := stepPlan b plan h stage p s
      let (events,final) := runPlan b plan h stage ps next
      (event::events,final)

def compileAndRun (b : Backend V) (source : List Statement) (h : HostArgs)
    (stage : Phased.Stage) (schedule : List Nat) (mem : Memory V) :
    Option (List SourceEvent × Wide.State V) := do
  let plan ← compile source
  if evalB h stage 0 0 0 plan.assertion then
    some (runPlan b plan h stage schedule (initialPlan b plan mem)) else none

end VeriTile.Bench.Repairs.SpinningLock.SourceReader
