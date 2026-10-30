import VeriTile.Triton.Float.ReductionProgramIO
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.FinCases

namespace VeriTile.Triton.FP.Structural.ReductionProgramRegression
noncomputable section

inductive Term where
  | raw (n : Nat)
  | zero
  | load (x : Term)
  | add (x y : Term)
  | cast (x : Term)
  deriving DecidableEq, Inhabited

def algebra (M : Algebra Term) : Algebra Term :=
  { M with
    literal := (fun _ _ _ => .zero)
    binary := (fun _ _ _ => Term.add)
    fp32Load := Term.load
    cast := (fun _ _ _ => Term.cast) }

def one : FP32AddPlan 1 := ⟨.input 0, by simp [FP32AddTree.leaves, List.ofFn_succ]⟩

def two : FP32AddPlan 2 :=
  ⟨.add (.zero) (.add (.input 0) (.input 1)),
    by simp [FP32AddTree.leaves, List.ofFn_succ]⟩

def initial (s : State Term) : State Term :=
  { s with mem := fun r a => if r = "input" then .mk .real (.raw a) else s.mem r a }

def single (M : Algebra Term) : FP32SumStage Term where
  backend := algebra M
  inputCount := 2
  outputCount := 1
  read := fun _ k => ("input", k.val)
  write := fun _ => ("out", 0)
  plans := fun _ => two

theorem exact_fp32_boundaries (M : Algebra Term) (s : State Term) :
    ((single M).run (initial s)).mem "out" 0 =
      .mk .real (.cast (.add .zero (.add (.load (.raw 0)) (.load (.raw 1))))) := by
  simp [FP32SumStage.run, FP32SumStage.value, single, two, FP32AddTree.eval,
    algebra, initial, List.ofFn_succ]

def aliasStage (M : Algebra Term) (sameDestination : Bool) : FP32SumStage Term where
  backend := algebra M
  inputCount := 1
  outputCount := 2
  read := fun o _ => ("input", if sameDestination then o.val else 0)
  write := fun o => ("input", if sameDestination then 0 else o.val)
  plans := fun _ => one

/-- The second output uses the stage-entry snapshot even though the first
store aliases its input. It does not recursively load the first result. -/
theorem reads_snapshot (M : Algebra Term) (s : State Term) :
    ((aliasStage M false).run (initial s)).mem "input" 1 =
      .mk .real (.cast (.load (.raw 0))) := by
  simp [FP32SumStage.run, FP32SumStage.value, aliasStage, one, FP32AddTree.eval,
    algebra, initial, List.ofFn_succ, State.write_other]

/-- If outputs alias, stores retain the declared increasing finite order. -/
theorem ordered_alias_writes (M : Algebra Term) (s : State Term) :
    ((aliasStage M true).run (initial s)).mem "input" 0 =
      .mk .real (.cast (.load (.raw 1))) := by
  simp [FP32SumStage.run, FP32SumStage.value, aliasStage, one, FP32AddTree.eval,
    algebra, initial, List.ofFn_succ]

def sourceWrite (M : Algebra Term) : Invocation Term where
  kernel := .mk [] ["input"] [
    .store .real [] (.region "input" (.constNat 0)) (.alg (.constFloat .real 0)) .none]
  backend := algebra M
  pids := fun _ => 0

def sourceFinal (M : Algebra Term) (s : State Term) : State Term :=
  ((sourceWrite M).initial s).write "input" 0 (.mk .real .zero)

theorem source_executes (M : Algebra Term) (s : State Term) :
    (sourceWrite M).run s = some (sourceFinal M s) := by
  simp [sourceWrite, Invocation.run, exec, run, step, evalExpr, evalOp_unfold,
    store, TileShape.allIndices, sourceFinal, algebra, Region.cast, ofFloat]

/-- A reduction stage consumes the actual successful source predecessor. -/
theorem consumes_source_predecessor (M : Algebra Term) (s : State Term) :
    ∃ final, runReductionStages
      [.invocation (sourceWrite M), .fp32Sum (aliasStage M false)] (initial s) = some final ∧
      final.mem "input" 1 = .mk .real (.cast (.load .zero)) := by
  refine ⟨(aliasStage M false).run (sourceFinal M (initial s)), ?_, ?_⟩
  · simp [runReductionStages, ReductionStage.run, source_executes]
  · simp [FP32SumStage.run, FP32SumStage.value, aliasStage, one, FP32AddTree.eval,
      algebra, sourceFinal, List.ofFn_succ, State.write_other]

theorem exact_exterior_frame (M : Algebra Term) (s : State Term) :
    ((single M).run s).mem "sentinel" 7 = s.mem "sentinel" 7 :=
  FP32SumStage.frame (single M) s _ _ (fun _ => Or.inl (by change ("sentinel" : RegionName) ≠ "out"; decide))

def empty (M : Algebra Term) : FP32SumStage Term where
  backend := algebra M
  inputCount := 999
  outputCount := 0
  read := fun o => Fin.elim0 o
  write := fun o => Fin.elim0 o
  plans := fun o => Fin.elim0 o

theorem empty_stage (M : Algebra Term) (s : State Term) :
    (empty M).run s = s ∧ (empty M).Safe (fun _ => 0) := by
  constructor
  · rfl
  · exact ⟨fun o => Fin.elim0 o, fun o => Fin.elim0 o⟩

theorem reduction_oob_rejected (M : Algebra Term) :
    ¬ (single M).Safe (fun r => if r = "out" then 1 else 0) := by
  intro safe
  have impossible := safe.1 (0 : Fin 1) (0 : Fin 2)
  simp [single] at impossible

def failed (M : Algebra Term) : Invocation Term where
  kernel := .mk [] [] [.assign .real [] "v" (.alg (.ref .real [] "missing"))]
  backend := algebra M
  clearRegisters := true
  pids := fun _ => 0

def failingIO (M : Algebra Term) : ReductionProgramIO Term Term where
  inputs := []
  outputs := []
  scratch := []
  buffers := []
  read := fun i => Fin.elim0 i
  inputMask := fun i => Fin.elim0 i
  write := fun o => Fin.elim0 o
  outputMask := fun o => Fin.elim0 o
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  stages := fun _ => [.invocation (failed M), .fp32Sum (empty M)]
  encode := id
  observe := fun o => Fin.elim0 o

theorem failed_source_cannot_be_hidden (M : Algebra Term) (s : State Term) :
    ¬ (failingIO M).Implements (fun _ o => Fin.elim0 o) := by
  intro claimed
  obtain ⟨_, final, ran, _⟩ := claimed (fun i => Fin.elim0 i) trivial s
    (fun i => Fin.elim0 i)
  simp [failingIO, runReductionStages, ReductionStage.run, failed,
    Invocation.run, Invocation.initial, bindRegisters, exec, run, step, evalExpr, evalOp_unfold] at ran

#axiomsClean exact_fp32_boundaries
#axiomsClean reads_snapshot
#axiomsClean ordered_alias_writes
#axiomsClean consumes_source_predecessor
#axiomsClean exact_exterior_frame
#axiomsClean empty_stage
#axiomsClean reduction_oob_rejected
#axiomsClean failed_source_cannot_be_hidden
end
end VeriTile.Triton.FP.Structural.ReductionProgramRegression
