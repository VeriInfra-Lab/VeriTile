import bench.tritonbench_g.int_scaled_matmul.Proofs.RealExecution
namespace VeriTile.Bench.Repairs.IntScaledMatmul.RealScaled
open VeriTile.Bench.TritonBenchG.IntScaledMatmul
open VeriTile.Triton FP.Structural FP.IntegerConversion
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

theorem loopExecution (rest : Algebra ℝ)
    (p : Parameters) (initial current : State ℝ) (k : Nat) (hk : k ≤ p.steps)
    (h : Invariant p initial k current) :
    ∃ final, FP.Structural.range (backend rest) "j" k p.steps 1 (body p) current = some final ∧
      Invariant p initial p.steps final := by
  have aux : ∀ n k (current : State ℝ), p.steps - k = n → k ≤ p.steps →
      Invariant p initial k current →
      ∃ final, FP.Structural.range (backend rest) "j" k p.steps 1 (body p) current = some final ∧
        Invariant p initial p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := bodyExecution rest p initial current k h
        have hi := bodyInvariant rest p initial current k h
        obtain ⟨final, he, hf⟩ := ih (p.steps - (k + 1)) (by omega)
          (k + 1) next.val rfl (by omega) hi
        refine ⟨final, ?_, hf⟩
        rw [FP.Structural.range, if_neg (by decide), if_pos hlt, next.property]
        exact he
      · have heq : k = p.steps := by omega
        subst k
        refine ⟨current, ?_, h⟩
        rw [FP.Structural.range, if_neg (by decide), if_neg (by omega)]
  exact aux _ _ _ rfl hk h

def epilogueExecution (rest : Algebra ℝ)
    (p : Parameters) (initial current : State ℝ)
    (h : Invariant p initial p.steps current) :
    {t : State ℝ // run (backend rest) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, Revision001.ScaledWord.epilogue, original, Revision001.ScaledWord.original, typedScaled, Revision001.ScaledWord.typedScaled, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.pidm, h.pidn, h.acc, h.memory, h.undefined, backend,
    wrap]
  rfl

def written (p : Parameters) (initial : State ℝ) : State ℝ :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then
      t.write p.output (outputOffset p initial i) (.mk .int (expected p initial i)) else t) initial

def Post (p : Parameters) (initial final : State ℝ) (values : CIndex p → Int) : Prop :=
  (∀ i,active p initial i → final.mem p.output (outputOffset p initial i)=.mk .int (values i)) ∧
  (∀ r o,(r≠p.output ∨ ∀ i,active p initial i → o≠outputOffset p initial i) →
    final.mem r o=initial.mem r o)
theorem epilogueMemory (rest : Algebra ℝ)
    (p : Parameters) (initial current : State ℝ)
    (h : Invariant p initial p.steps current) :
    (epilogueExecution rest p initial current h).val.mem = (written p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputRow, outputColumn, expected, scaled, wrap,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]

theorem executionMemory (rest : Algebra ℝ)
    (p : Parameters) (initial : State ℝ) :
    ∃ final, FP.Structural.exec (backend rest) (source p) initial = some final ∧
      final.mem = (written p initial).mem := by
  let start := preludeExecution rest p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution rest p initial start.val 0 (by omega)
    (preludeInvariant rest p initial)
  let finish := epilogueExecution rest p initial mid hi
  refine ⟨finish.val, ?_, epilogueMemory rest p initial mid hi⟩
  change run (backend rest) (source p).surfaceBody initial = some finish.val
  rw [show (source p).surfaceBody = prelude p ++ [.forRange "j" 0 p.steps 1 (body p)] ++ epilogue p from by simpa only [source,SourcePreservation.scaled] using (show (Revision001.ScaledWord.source p).surfaceBody = prelude p ++ [.forRange "j" 0 p.steps 1 (body p)] ++ epilogue p from rfl),
    run_append, run_append, start.property]
  have hsingle (s : State ℝ) :
      run (backend rest) [.forRange "j" 0 p.steps 1 (body p)] s =
        FP.Structural.range (backend rest) "j" 0 p.steps 1 (body p) s := by simp [run,step]
  simp only [Option.bind_some, hsingle, hm]
  exact finish.property

theorem correctness (rest : Algebra ℝ)
    (p : Parameters) (initial : State ℝ)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend rest) (source p) initial = some final ∧
      Post p initial final (expected p initial) := by
  obtain ⟨final,he,hm⟩ := executionMemory rest p initial
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o hframe
    rw [hm]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hframe.imp id (fun h i _ => h i)) _

#axiomsClean executionMemory
#axiomsClean correctness
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RealScaled
