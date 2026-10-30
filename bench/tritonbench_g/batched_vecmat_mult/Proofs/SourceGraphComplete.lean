import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphExecution

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨Expr.zero⟩
attribute [local irreducible] treeValue

theorem loop_correct (rest : Algebra Word) (plans : PlanSelector)
    (p : Parameters) (initial : State Word) (fuel c : Nat) (current : State Word)
    (hle : c ≤ p.K / p.BK) (hbound : p.K / p.BK - c ≤ fuel)
    (h : Invariant plans p initial c current) :
    ∃ final, FP.Structural.range (engine rest plans) "k_index" c (p.K / p.BK) 1
      (body p) current = some final ∧ Invariant plans p initial (p.K / p.BK) final := by
  induction fuel generalizing c current with
  | zero =>
    have he : c = p.K / p.BK := by omega
    subst c
    refine ⟨current, ?_, h⟩
    rw [FP.Structural.range]
    simp
  | succ fuel ih =>
    by_cases hc : c < p.K / p.BK
    · let next := bodyExecution rest plans p initial current c h
      obtain ⟨final, he, hf⟩ := ih (c+1) next.val (by omega) (by omega)
        (bodyInvariant rest plans p initial current c h)
      refine ⟨final, ?_, hf⟩
      rw [FP.Structural.range, if_neg (by decide), if_pos hc, next.property]
      exact he
    · have he : c = p.K / p.BK := by omega
      subst c
      refine ⟨current, ?_, h⟩
      rw [FP.Structural.range]
      simp

def tailExecution (rest : Algebra Word) (plans : PlanSelector)
    (p : Parameters) (initial current : State Word)
    (h : Invariant plans p initial (p.K / p.BK) current) :
    {t : State Word // run (engine rest plans) (tail p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [tail, source, TypedSource.original, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    store, h.offsets, h.carry, Region.cast]
  rfl

theorem fold_write_memory (p : Parameters) (initial s : State Word)
    (values : TileIndex [p.BM,p.BN] → Word) (xs : List (TileIndex [p.BM,p.BN])) :
    (xs.foldl (fun t i => t.write p.output (outputOffset p initial i) (.mk .real (values i))) s).mem =
      xs.foldl (fun mem i => fun r o => if r = p.output ∧ o = outputOffset p initial i then
        .mk .real (values i) else mem r o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons, ih]
    rfl

theorem tail_memory (rest : Algebra Word) (plans : PlanSelector)
    (p : Parameters) (initial current : State Word)
    (h : Invariant plans p initial (p.K / p.BK) current) :
    (tailExecution rest plans p initial current h).val.mem = expectedMemory plans p initial := by
  simp only [tailExecution]
  change ((TileShape.allIndices [p.BM,p.BN]).foldl
    (fun t i => t.write p.output (outputOffset p initial i)
      (.mk .real (accumulated plans p initial (p.K / p.BK) i))) current).mem = _
  rw [fold_write_memory, h.memory]
  rfl

theorem complete_correct (rest : Algebra Word) (plans : PlanSelector)
    (p : Parameters) (initial : State Word) :
    ∃ final, FP.Structural.exec (engine rest plans) (source p) initial = some final ∧
      final.mem = expectedMemory plans p initial ∧
      (∀ r o, Outside p initial r o → final.mem r o = initial.mem r o) := by
  let pre := prologueExecution rest plans p initial
  have hp : Invariant plans p initial 0 pre.val := prologueInvariant rest plans p initial
  obtain ⟨last, hl, hi⟩ := loop_correct rest plans p initial (p.K / p.BK) 0 pre.val
    (Nat.zero_le _) (by simp) hp
  let final := tailExecution rest plans p initial last hi
  refine ⟨final.val, ?_, tail_memory rest plans p initial last hi, ?_⟩
  · change run (engine rest plans) (source p).surfaceBody initial = _
    rw [source_split, run_append, run_append, pre.property]
    simp only [Option.bind_some, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      bind, Option.bind]
    rw [hp.blocks]
    simp only [hl]
    exact final.property
  · intro r o ho
    rw [tail_memory]
    exact expectedMemory_frame plans p initial r o ho

#axiomsClean loop_correct
#axiomsClean tailExecution
#axiomsClean fold_write_memory
#axiomsClean tail_memory
#axiomsClean complete_correct
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
