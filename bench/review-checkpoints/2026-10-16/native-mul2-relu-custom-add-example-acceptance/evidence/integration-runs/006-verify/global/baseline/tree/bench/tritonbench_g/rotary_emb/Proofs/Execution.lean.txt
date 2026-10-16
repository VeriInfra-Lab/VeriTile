import bench.tritonbench_g.rotary_emb.Proofs.ExecutionK
import bench.tritonbench_g.rotary_emb.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FullSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000

/-- The complete source is composed from successful execution of its consecutive
blocks. Every block executes its actual statements, including Q stores before
K loads and both trigonometric reloads. -/
def original_run (flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model) (program Bool.false flip).surfaceBody
      (initial flip) = some t} := by
  refine ⟨(kstores_run flip).val, ?_⟩
  have hsplit : (program Bool.false flip).surfaceBody =
      (program Bool.false flip).surfaceBody.take 6 ++
      ((program Bool.false flip).surfaceBody.drop 6).take 4 ++
      ((program Bool.false flip).surfaceBody.drop 10).take 6 ++
      ((program Bool.false flip).surfaceBody.drop 16).take 2 ++
      ((program Bool.false flip).surfaceBody.drop 18).take 2 ++
      ((program Bool.false flip).surfaceBody.drop 20).take 4 ++
      ((program Bool.false flip).surfaceBody.drop 24).take 6 ++
      ((program Bool.false flip).surfaceBody.drop 30).take 2 ++
      ((program Bool.false flip).surfaceBody.drop 32).take 2 := by
    cases flip <;> rfl
  conv_lhs => rw [hsplit]
  simp only [run_append, (prefix_run flip).property, (addresses_run flip).property,
    (qloads_run flip).property, (qarithmetic_run flip).property, (qstores_run flip).property,
    (kaddresses_run flip).property, (kloads_run flip).property,
    (karithmetic_run flip).property, (kstores_run flip).property, Option.bind_some]

private theorem program_variants_run (flip : Bool) :
    run (engine model) (program Bool.false flip).surfaceBody (initial flip) =
      run (engine model) (program Bool.true flip).surfaceBody (initial flip) := by
  unfold program
  rw [FPProof.decomposition, FPProof.decomposition]
  exact FPProof.prefix_run model (initial flip) 2 _

/-- Two sequences, two heads, two pairs, nonzero program IDs, independent
nonunit strides and both sequence/head padding are present in the full run. -/
def numeric_run (v flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model) (program v flip).surfaceBody
      (initial flip) = some t} := by
  refine ⟨(original_run flip).val, ?_⟩
  cases v
  · exact (original_run flip).property
  · rw [← program_variants_run flip]
    exact (original_run flip).property

def qCell (flip : Bool) (x : ℝ) : Cell (WithBot ℝ) :=
  if flip then .mk .real (some x) else .mk .fp16 (ofFloat .fp16 (some x))

def kCell (flip : Bool) (x : ℝ) : Cell (WithBot ℝ) :=
  if flip then .mk .fp16 (ofFloat .fp16 (some x)) else .mk .real (some x)

/-- Odd trigonometric values differ from the even values: the Q odd result is
25, not the 11 produced by reusing the even trigonometric inputs. All four store
families execute, including the second K head; tail and stride-gap cells persist. -/
theorem numeric_cells (v flip : Bool) :
    (numeric_run v flip).val.mem "Q" 220 = qCell flip (-7) ∧
    (numeric_run v flip).val.mem "Q" 222 = qCell flip 25 ∧
    (numeric_run v flip).val.mem "Q" 224 = qCell flip (-7) ∧
    (numeric_run v flip).val.mem "K" 440 = kCell flip (-13) ∧
    (numeric_run v flip).val.mem "K" 443 = kCell flip 79 ∧
    (numeric_run v flip).val.mem "K" 469 = kCell flip 79 ∧
    (numeric_run v flip).val.mem "Q" 230 = qCell flip 7 ∧
    (numeric_run v flip).val.mem "Q" 320 = qCell flip 7 ∧
    (numeric_run v flip).val.mem "Q" 221 = qCell flip 7 ∧
    (numeric_run v flip).val.mem "Sin" 104 = (initial flip).mem "Sin" 104 := by
  simp only [numeric_run, original_run]
  cases flip
  all_goals simp (maxSteps := 200000) [kstores_run, karithmetic_run, kloads_run,
    kaddresses_run, qstores_run, qarithmetic_run, qloads_run, addresses_run, prefix_run,
    initial, model, payload, qCell, kCell,
    FP.Structural.store, State.write, State.setReg, Region.cast, Tile.scalar, -Tile.scalar_eta,
    Function.comp_def, TileShape.allIndices, TileShape.dropInsertedIndex,
    TileShape.insertAxis, List.finRange_succ, List.foldl, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv]
  all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map,
    Option.bind, bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
    Function.comp_def, TileShape.dropInsertedIndex, TileShape.insertAxis,
    List.finRange_succ, List.foldl, List.flatMap]
  -- Fin instances retain their original, definitionally equal shape bounds.
  -- Reduce only those closed index projections before the final real arithmetic.
  all_goals run_tac
    let goal ← Lean.Elab.Tactic.getMainGoal
    let target ← goal.getType
    let normalized ← Lean.Meta.transform target (pre := fun e => do
      if e.isAppOfArity ``VeriTile.Triton.TileShape.dropInsertedIndex 4 then
        return .done (← Lean.Meta.reduce e)
      return .continue)
    let next ← goal.replaceTargetDefEq normalized
    Lean.Elab.Tactic.replaceMainGoal [next]
  all_goals norm_num [Option.map₂, Option.map, Option.bind, WithBot.some_eq_coe]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution
