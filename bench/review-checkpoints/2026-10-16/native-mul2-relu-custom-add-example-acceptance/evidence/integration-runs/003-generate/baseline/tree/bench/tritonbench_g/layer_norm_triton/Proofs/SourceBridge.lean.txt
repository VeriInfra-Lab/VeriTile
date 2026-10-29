import bench.tritonbench_g.layer_norm_triton.Proofs.GeneratedSource

namespace VeriTile.Bench.TritonBenchG.LayerNormLock.Frontend
open VeriTile.Triton
open FP.Equational (ReductionPlan)

/-- Independently written expected source graph, checked against generated AST
lowering. This is not a numerical value or an assumed equality of executions. -/
def originalCompiled : Compiled where
  forward := ⟨.x, .div (.slot .meanSum) .denominator,
    .mul (.mask (.sub .x (.slot .mean))) (.mask (.sub .x (.slot .mean))),
    .div (.slot .varSum) .denominator,
    .div .one (.sqrt (.add (.slot .variance) (.slot .eps))),
    .add (.mul (.mul (.sub .x (.slot .mean)) (.slot .rstd)) (.promoteW .w)) (.promoteBias .bias)⟩
  backwardProgram := backwardProgram
  lock := ⟨0,1,1,0,1,0⟩

theorem compile_original : compile originalTypedSource = some originalCompiled := by decide +kernel

theorem forward_original (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (eps : V) (p : Nat) (offsets : List I32) :
    executeForward originalCompiled.forward b c plan mem undef eps p offsets =
      forwardSource b c plan mem undef eps p offsets := by
  rfl

theorem rows_original (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (p : Nat) :
    compiledRows originalCompiled b c plan mem undef p = backwardSource b c plan mem undef p := rfl

theorem decode_original (c : Config) (p : Nat) (s : Control) :
    decodePlan originalCompiled.lock c p s = decode c p s := by
  by_cases hp : p ≥ c.grid
  · simp [decodePlan, decode, hp]
  · cases h : s.pc p <;> simp [decodePlan, decode, hp, h, originalCompiled]

theorem integer_original (e : Event) (words : Int → I32) :
    integerPlan originalCompiled.lock e words = integerEffect e words := by
  cases e <;> rfl

theorem control_step_original (c : Config) (p : Nat) (s : Control) :
    controlStepPlan originalCompiled.lock c p s = controlStep c p s := by
  simp only [controlStepPlan, controlStep, decode_original, integer_original]

theorem source_step_original (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (p : Nat) (s : State V) :
    sourceStep originalCompiled.lock b c rows undef p s = step b c rows undef p s := by
  simp only [sourceStep, step, control_step_original]

theorem source_run_original (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (schedule : List Nat) (s : State V) :
    sourceRun originalCompiled.lock b c rows undef schedule s = run b c rows undef schedule s := by
  induction schedule generalizing s with
  | nil => rfl
  | cons p ps ih => simp only [sourceRun, run, source_step_original, ih]

def compileForward (source : TypedSource) (b : Backend V) (c : Config)
    (plan : ReductionPlan c.block) (mem : Memory V) (undef : Undef V) (eps : V) (p fuel : Nat) : Option (Memory V) := do
  let graph ← compile source
  let offsets ← rangeTrace c.n (BitVec.ofNat 32 c.block) 0 fuel
  return writeAll (forwardWrites b c p offsets (executeForward graph.forward b c plan mem undef eps p offsets)) mem

def compileBackward (source : TypedSource) (b : Backend V) (c : Config)
    (plan : ReductionPlan c.block) (inputs : Memory V) (undef : Undef V)
    (mem : Shared V) (words : Int → I32) (schedule : List Nat) : Option (List Event × State V) := do
  let graph ← compile source
  let rows := compiledRows graph b c plan inputs undef
  return sourceRun graph.lock b c rows undef schedule (initial rows mem words)

def compileFinal (source : TypedSource) (b : Backend V) (c : Config) (mem : Memory V)
    (m : I32) (bm bn pid : Nat) (plan : ReductionPlan bm) (fuel : Nat) : Option (Memory V) := do
  let _graph ← compile source
  finalKernel b c mem m bm bn pid plan fuel

theorem original_forward_execution (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (eps : V) (p fuel : Nat) :
    compileForward originalTypedSource b c plan mem undef eps p fuel =
      forwardKernel b c plan mem undef eps p fuel := by
  simp only [compileForward, compile_original, Option.bind_eq_bind, Option.pure_def, Option.bind_some, forward_original, forwardKernel, Option.map_eq_bind, Function.comp_def]

theorem original_backward_execution (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (inputs : Memory V) (undef : Undef V) (mem : Shared V) (words : Int → I32) (schedule : List Nat) :
    compileBackward originalTypedSource b c plan inputs undef mem words schedule =
      some (run b c (backwardSource b c plan inputs undef) undef schedule
        (initial (backwardSource b c plan inputs undef) mem words)) := by
  have hr : compiledRows originalCompiled b c plan inputs undef = backwardSource b c plan inputs undef := rfl
  simp only [compileBackward, compile_original, Option.bind_eq_bind, Option.pure_def, Option.bind_some, hr, source_run_original]

theorem original_final_execution (b : Backend V) (c : Config) (mem : Memory V)
    (m : I32) (bm bn pid : Nat) (plan : ReductionPlan bm) (fuel : Nat) :
    compileFinal originalTypedSource b c mem m bm bn pid plan fuel = finalKernel b c mem m bm bn pid plan fuel := by
  simp only [compileFinal, compile_original, Option.bind_eq_bind, Option.pure_def, Option.bind_some]

#axiomsClean compile_original
#axiomsClean forward_original
#axiomsClean source_run_original
#axiomsClean original_forward_execution
#axiomsClean original_backward_execution
#axiomsClean original_final_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock.Frontend
