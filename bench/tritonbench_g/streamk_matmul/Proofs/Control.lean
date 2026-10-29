import bench.tritonbench_g.streamk_matmul.Proofs.Types
namespace VeriTile.Bench.TritonBenchG.StreamK

/-- Full source small-step decoder. Pure bindings are simultaneous per tile;
loads/stores/atomic-adds are individually interleavable lane operations. CAS
reads the current lock and writes its requested replacement before branching. -/
def decode (c : Config) (grid p : Nat) (locks : Address→I32) (s : Control) : Event×Control :=
 if p≥grid || s.fault then (.silent,s) else
 match s.todo with
 | [] => (.silent,s)
 | cmd::rest =>
  let next := {s with todo := rest}
  let fault := (Event.silent,{next with fault := true})
  match cmd with
  | .pass => (.silent,next)
  | .integer name shape val =>
    let values := fun i=>evalI c p shape i s val
    if (List.range (c.count shape)).all (fun i=>(values i).isSome) then
      (.silent,{next with ints := put s.ints name values}) else fault
  | .pointer name shape val =>
    let values := fun i=>evalPointer c p shape i s val
    if (List.range (c.count shape)).all (fun i=>(values i).isSome) then
      (.silent,{next with pointers := put s.pointers name values}) else fault
  | .zero name _ => (.reset p name,next)
  | .load name shape ptr => (.silent,{next with todo := .loadLane name shape ptr 0::rest})
  | .loadLane name shape ptr lane =>
    if lane<c.count shape then
      match evalPointer c p shape lane s ptr with
      | none => fault
      | some a => (.load p name lane a,{next with todo := .loadLane name shape ptr (lane+1)::rest})
    else (.silent,next)
  | .mac name a b => (.mac p name a b,next)
  | .half name v => (.half p name v,next)
  | .store shape ptr v => (.silent,{next with todo := .storeLane false shape ptr v 0::rest})
  | .atomicAdd shape ptr v => (.silent,{next with todo := .storeLane true shape ptr v 0::rest})
  | .storeLane atomic shape ptr v lane =>
    if lane<c.count shape then
      match evalPointer c p shape lane s ptr with
      | none => fault
      | some a => (if atomic then .atomicAdd p v lane a else .store p v lane a,
          {next with todo := .storeLane atomic shape ptr v (lane+1)::rest})
    else (.silent,next)
  | .exchange ptr v =>
    match evalPointer c p .scalar 0 s ptr, evalI c p .scalar 0 s v with
    | some a,some v => (.exchange a v,next)
    | _,_ => fault
  | .branch test yes no =>
    match evalB c p s test with
    | none => fault
    | some b => (.silent,{next with todo := (if b then yes else no)++rest})
  | .whileLoop test body =>
    let tail := body++cmd::rest
    match test with
    | .pure b =>
      match evalB c p s b with
      | none => fault
      | some b => (.silent,{next with todo := if b then tail else rest})
    | .casNe ptr cmp rep expected =>
      match evalPointer c p .scalar 0 s ptr,evalI c p .scalar 0 s cmp,
        evalI c p .scalar 0 s rep,evalI c p .scalar 0 s expected with
      | some a,some cmp,some rep,some expected =>
        (.cas a cmp rep,{next with todo := if locks a != expected then tail else rest})
      | _,_,_,_ => fault
  | .forLoop index start stop body =>
    match evalI c p .scalar 0 s start,evalI c p .scalar 0 s stop with
    | some lo,some hi => (.silent,{next with
        ints := put s.ints index (fun _=>some lo)
        todo := .forNext index hi body::rest})
    | _,_ => fault
  | .forNext index stop body =>
    match s.ints index 0 with
    | none => fault
    | some current =>
      (.silent,{next with todo := if current.slt stop then
        body++[.integer index .scalar (.add (.get index .zero) (.lit 1)),cmd]++rest else rest})

def lockEffect (e : Event) (locks : Address→I32) : Address→I32 :=
 match e with
 | .exchange a v => put locks a v
 | .cas a cmp rep => put locks a (if locks a==cmp then rep else locks a)
 | _ => locks

structure ControlState where
 locks : Address→I32
 programs : Nat→Control

def initialControl (source : List Command) : Control := ⟨source,fun _ _=>none,fun _ _=>none,false⟩
def controlInitial (source : List Command) (locks : Address→I32) : ControlState :=
 ⟨locks,fun _=>initialControl source⟩
def initial (source : List Command) (b : Backend V) (mem : Memory V) : State V :=
 ⟨mem,fun _=>initialControl source,fun _ _ _=>b.zero⟩
def controlStep (c : Config) (grid p : Nat) (s : ControlState) : Event×ControlState :=
 let (e,ctl):=decode c grid p s.locks (s.programs p)
 (e,⟨lockEffect e s.locks,put s.programs p ctl⟩)
def controlRun (c : Config) (grid : Nat) : List Nat→ControlState→List Event×ControlState
 | [],s => ([],s)
 | p::ps,s =>
   let (e,s'):=controlStep c grid p s
   let (es,s''):=controlRun c grid ps s'
   (e::es,s'')
def controlTrace (source : List Command) (c : Config) (grid : Nat)
    (schedule : List Nat) (locks : Address→I32) : List Event :=
 (controlRun c grid schedule (controlInitial source locks)).1

def step (c : Config) (b : Backend V) (grid p : Nat) (s : State V) : Event×State V :=
 let (e,ctl):=decode c grid p s.mem.locks (s.control p)
 (e,{execute c b e s with control := put s.control p ctl})
def run (c : Config) (b : Backend V) (grid : Nat) : List Nat→State V→List Event×State V
 | [],s => ([],s)
 | p::ps,s =>
   let (e,s'):=step c b grid p s
   let (es,s''):=run c b grid ps s'
   (e::es,s'')
def projectControl (s : State V) : ControlState := ⟨s.mem.locks,s.control⟩

theorem step_control (c : Config) (b : Backend V) (grid p : Nat) (s : State V) :
 controlStep c grid p (projectControl s)=((step c b grid p s).1,projectControl (step c b grid p s).2) := by
 simp only [controlStep,step,projectControl]
 cases h:decode c grid p s.mem.locks (s.control p) with
 | mk e ctl => cases e <;> rfl

theorem run_control (c : Config) (b : Backend V) (grid : Nat) (schedule : List Nat) (s : State V) :
 controlRun c grid schedule (projectControl s)=((run c b grid schedule s).1,projectControl (run c b grid schedule s).2) := by
 induction schedule generalizing s with
 | nil => rfl
 | cons p ps ih =>
   simp only [controlRun,run]
   rw [step_control c b grid p s]
   simp only [ih (step c b grid p s).2]

theorem trace_independent (source : List Command) (c : Config) (b : Backend V)
 (grid : Nat) (schedule : List Nat) (mem : Memory V) :
 (run c b grid schedule (initial source b mem)).1=controlTrace source c grid schedule mem.locks := by
 exact (congrArg Prod.fst (run_control c b grid schedule (initial source b mem))).symm

#axiomsClean run_control
#axiomsClean trace_independent
end VeriTile.Bench.TritonBenchG.StreamK
