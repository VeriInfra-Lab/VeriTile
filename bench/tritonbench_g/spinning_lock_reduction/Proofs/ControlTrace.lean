import bench.tritonbench_g.spinning_lock_reduction.Proofs.FrontendChecks

/-! An independent scheduler carrying ONLY program counters and integer locks.
There is no FP backend, no P/C memory, no accumulator, and no call to execute or
runPlan. This runner supplies the final numerical specification's event trace.
-/
namespace VeriTile.Bench.TritonBenchG.SpinningLock.Frontend

structure ControlState where
  locks : Int → I32
  control : Nat → Control

def projectControl (s : State V) : ControlState := ⟨s.mem.locks, s.control⟩

/-- Reuse only the pure source control decoder. Its temporary Unit fields are
unobservable: it reads PC/locks and Config expressions, never floating data. -/
def decodeControl (plan : Plan) (c : Config) (p : Nat) (s : ControlState) : Event × Control :=
  decodePlan plan c p
    ({ mem := ⟨fun _ => (), fun _ => (), s.locks⟩,
       control := s.control, acc := fun _ _ => () } : State Unit)

def stepControl (plan : Plan) (c : Config) (p : Nat) (s : ControlState) : Event × ControlState :=
  let (e, ctl) := decodeControl plan c p s
  let locks := match e with
    | .release a => put s.locks a 1
    | _ => s.locks
  (e, ⟨locks, put s.control p ctl⟩)

def runControl (plan : Plan) (c : Config) : List Nat → ControlState → List Event × ControlState
  | [], s => ([], s)
  | p :: ps, s =>
    let (e, s') := stepControl plan c p s
    let (es, final) := runControl plan c ps s'
    (e :: es, final)

def controlInitial (plan : Plan) (locks : Int → I32) : ControlState :=
  ⟨locks, fun _ => { round := plan.start }⟩

/-- The reference trace is computable before interpreting ANY floating value. -/
def controlTrace (plan : Plan) (c : Config) (schedule : List Nat) (locks : Int → I32) : List Event :=
  (runControl plan c schedule (controlInitial plan locks)).1

theorem decode_control_projection (plan : Plan) (c : Config) (p : Nat) (s : State V) :
    decodeControl plan c p (projectControl s) = decodePlan plan c p s := rfl

theorem step_control_projection (plan : Plan) (b : Backend V) (c : Config)
    (p : Nat) (s : State V) :
    stepControl plan c p (projectControl s) =
      ((stepPlan plan b c p s).1, projectControl (stepPlan plan b c p s).2) := by
  simp only [stepControl, stepPlan, decode_control_projection]
  cases h : decodePlan plan c p s with
  | mk e ctl => cases e <;> rfl

theorem run_control_projection (plan : Plan) (b : Backend V) (c : Config)
    (schedule : List Nat) (s : State V) :
    runControl plan c schedule (projectControl s) =
      ((runPlan plan b c schedule s).1, projectControl (runPlan plan b c schedule s).2) := by
  induction schedule generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [runControl, runPlan, step_control_projection plan b c p s,
      ih (stepPlan plan b c p s).2]

theorem control_trace_independent_of_values (plan : Plan) (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) :
    (runPlan plan b c schedule (initialPlan plan b mem)).1 =
      controlTrace plan c schedule mem.locks := by
  have h := congrArg Prod.fst
    (run_control_projection plan b c schedule (initialPlan plan b mem))
  exact h.symm

/-- Even two different carrier types/backends/memories have identical control
traces whenever their integer lock maps agree. No equality of P/C or numeric
results is required. This is proved, not a theorem premise about executions. -/
theorem arbitrary_backends_same_events (plan : Plan) (b : Backend V) (d : Backend W)
    (c : Config) (schedule : List Nat) (mem : Memory V) (other : Memory W)
    (hl : mem.locks = other.locks) :
    (runPlan plan b c schedule (initialPlan plan b mem)).1 =
      (runPlan plan d c schedule (initialPlan plan d other)).1 := by
  rw [control_trace_independent_of_values, control_trace_independent_of_values, hl]

theorem original_control_trace (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) :
    (run b c schedule (initial b mem)).1 = controlTrace originalPlan c schedule mem.locks := by
  simpa only [initial_original, run_original] using
    control_trace_independent_of_values originalPlan b c schedule mem

#axiomsClean decode_control_projection
#axiomsClean step_control_projection
#axiomsClean run_control_projection
#axiomsClean control_trace_independent_of_values
#axiomsClean arbitrary_backends_same_events
#axiomsClean original_control_trace
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock.Frontend
