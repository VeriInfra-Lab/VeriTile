import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FP64Memory

/-! Programs update actual tagged logical memory. Every lane reads the fixed
pre-program snapshot; successful writes update a separate working memory in
lane order. Errors return no final memory, a mathematical transaction policy
that does not model native poison, traps or partially visible stores. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.FP64Memory
open FP64TypedCore
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

def blockSize : Nat := 1024
def offset (pid lane : Nat) : Nat := pid * blockSize + lane

def laneOutcome (b : Numerics) (p : Parameters xD yD) (snapshot : Memory) (index : Nat) : Outcome xD :=
  runLane b xD yD (decide (index < p.n))
    (readCell xD (snapshot p.x index)) (readCell yD (snapshot p.y index))

inductive Fault where
  | inputUnavailable (index : Nat)
  | conversionUndefined (index : Nat)
  deriving DecidableEq, Repr

/-- There is no default value on an unavailable active input. Masked lanes
leave the working memory unchanged without inspecting either typed load. -/
def runLanes (b : Numerics) (p : Parameters xD yD) (snapshot : Memory) (pid : Nat) :
    List Nat → Memory → Except Fault Memory
  | [], working => .ok working
  | lane :: lanes, working =>
    match laneOutcome b p snapshot (offset pid lane) with
    | .masked => runLanes b p snapshot pid lanes working
    | .loadUnavailable => .error (.inputUnavailable (offset pid lane))
    | .conversionUndefined => .error (.conversionUndefined (offset pid lane))
    | .write v => runLanes b p snapshot pid lanes
        (writeCell working p.output (offset pid lane) (some (.mk xD v)))

def runProgram (b : Numerics) (p : Parameters xD yD) (pid : Nat) (initial : Memory) : Except Fault Memory :=
  runLanes b p initial pid (List.range blockSize) initial

theorem lane_masked (b : Numerics) (p : Parameters xD yD) (snapshot : Memory)
    (index : Nat) (inactive : ¬index < p.n) : laneOutcome b p snapshot index = .masked := by
  simp [laneOutcome, inactive, runLane]

theorem lane_active (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (snapshot : Memory) (inputs : InputWords p xs ys snapshot)
    (index : Nat) (active : index < p.n) :
    laneOutcome b p snapshot index =
      match result b xD yD (xs index) (ys index) with
      | some v => .write v | none => .conversionUndefined := by
  unfold laneOutcome
  rw [(inputs index active).1, (inputs index active).2]
  simp only [active, decide_true, read_cell_same]
  exact active_lane b xD yD (xs index) (ys index)

theorem lane_unavailable (b : Numerics) (p : Parameters xD yD) (snapshot : Memory)
    (index : Nat) (active : index < p.n)
    (missing : readCell xD (snapshot p.x index) = none ∨ readCell yD (snapshot p.y index) = none) :
    laneOutcome b p snapshot index = .loadUnavailable := by
  unfold laneOutcome
  simp only [active, decide_true]
  rcases missing with hx | hy
  · rw [hx]; exact missing_x b xD yD _
  · rw [hy]; exact missing_y b xD yD _

theorem lanes_input_error (b : Numerics) (p : Parameters xD yD) (snapshot working : Memory)
    (pid lane : Nat) (lanes : List Nat) (active : offset pid lane < p.n)
    (missing : readCell xD (snapshot p.x (offset pid lane)) = none ∨
      readCell yD (snapshot p.y (offset pid lane)) = none) :
    runLanes b p snapshot pid (lane :: lanes) working = .error (.inputUnavailable (offset pid lane)) := by
  rw [runLanes, lane_unavailable b p snapshot (offset pid lane) active missing]

theorem lanes_conversion_error (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (snapshot working : Memory) (inputs : InputWords p xs ys snapshot)
    (pid lane : Nat) (lanes : List Nat) (active : offset pid lane < p.n)
    (undefined : result b xD yD (xs (offset pid lane)) (ys (offset pid lane)) = none) :
    runLanes b p snapshot pid (lane :: lanes) working = .error (.conversionUndefined (offset pid lane)) := by
  rw [runLanes, lane_active b p xs ys snapshot inputs (offset pid lane) active, undefined]

/-- The current memory evolves through every actual store, while all input
reads retain the source program's pre-store snapshot. -/
theorem lanes_success (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (snapshot : Memory) (inputs : InputWords p xs ys snapshot)
    (defined : OutputDefined b p xs ys) (pid : Nat) (lanes : List Nat) (working : Memory) :
    runLanes b p snapshot pid lanes working =
      .ok (referenceIndices b p xs ys (lanes.map (offset pid)) working) := by
  induction lanes generalizing working with
  | nil => simp [runLanes]
  | cons lane lanes ih =>
    rw [runLanes]
    by_cases active : offset pid lane < p.n
    · obtain ⟨v, hv⟩ := defined_value b p xs ys defined (offset pid lane) active
      rw [lane_active b p xs ys snapshot inputs (offset pid lane) active, hv]
      dsimp only
      rw [ih]
      simp only [List.map_cons, reference_cons, active, ↓reduceIte]
      simp [outputCell, hv]
    · rw [lane_masked b p snapshot (offset pid lane) active, ih]
      simp [reference_cons, active]

theorem program_success (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (initial : Memory) (inputs : InputWords p xs ys initial)
    (defined : OutputDefined b p xs ys) (pid : Nat) :
    runProgram b p pid initial =
      .ok (referenceIndices b p xs ys ((List.range blockSize).map (offset pid)) initial) :=
  lanes_success b p xs ys initial inputs defined pid (List.range blockSize) initial

theorem lanes_all_masked (b : Numerics) (p : Parameters xD yD) (snapshot working : Memory)
    (pid : Nat) (lanes : List Nat) (inactive : ∀ lane ∈ lanes, ¬offset pid lane < p.n) :
    runLanes b p snapshot pid lanes working = .ok working := by
  induction lanes generalizing working with
  | nil => rfl
  | cons lane lanes ih =>
    rw [runLanes, lane_masked b p snapshot (offset pid lane) (inactive lane (by simp))]
    exact ih working (fun l hl => inactive l (by simp [hl]))

theorem empty_program (b : Numerics) (p : Parameters xD yD) (initial : Memory)
    (pid : Nat) (empty : p.n = 0) : runProgram b p pid initial = .ok initial := by
  apply lanes_all_masked
  intro lane _
  simp [empty]

#specNonCircular reference avoiding [runLanes, runProgram, laneOutcome, source, Expr.eval, runLane]
#specNonCircular OutputDefined avoiding [runLanes, runProgram, laneOutcome, source, Expr.eval, runLane]
#axiomsClean lane_masked
#axiomsClean lane_active
#axiomsClean lane_unavailable
#axiomsClean lanes_input_error
#axiomsClean lanes_conversion_error
#axiomsClean lanes_success
#axiomsClean program_success
#axiomsClean lanes_all_masked
#axiomsClean empty_program
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.FP64Memory
