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
    (p : Parameters) (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (cover : p.inner≤p.steps*p.blockInner)
    (initial current : State ℝ) (k : Nat) (hk : k≤p.steps)
    (h : Invariant p initial k current) :
    ∃ final, FP.Structural.rangeInt (backend rest) "__rev_k" (Revision001.ScaledWord.mirror p k) 0 p.blockInner
      (body p) current=some final ∧ OutputReady p initial final := by
  have finished (k : Nat) (current : State ℝ) (hk : k≤p.steps)
      (h : Invariant p initial k current) (stopped : p.inner≤k*p.blockInner) : OutputReady p initial current := by
    have equal : accumulation p initial p.steps=accumulation p initial k := by
      funext i
      exact stopped_accumulation p initial heven k hk stopped i
    exact ⟨h.memory,h.undefined,h.pidm,h.pidn,by simpa only [equal] using h.acc,h.rows,h.columns⟩
  have aux : ∀ n k (current : State ℝ),p.steps-k=n → k≤p.steps →
      Invariant p initial k current →
      ∃ final, FP.Structural.rangeInt (backend rest) "__rev_k" (Revision001.ScaledWord.mirror p k) 0 p.blockInner
        (body p) current=some final ∧ OutputReady p initial final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih=>
      intro k current hn hk h
      by_cases zero : p.blockInner=0
      · refine ⟨current,?_,finished k current hk h ?_⟩
        · rw [FP.Structural.rangeInt,if_pos zero]
        · simpa [zero] using cover
      by_cases live : Revision001.ScaledWord.mirror p k<0
      · have active : k*p.blockInner<p.inner := by
          have hx : (k*p.blockInner:Int)<(p.inner:Int) := by
            simpa only [Revision001.ScaledWord.mirror,sub_lt_zero] using live
          exact_mod_cast hx
        have less : k<p.steps := by
          by_contra hn
          have le:=Nat.mul_le_mul_right p.blockInner (Nat.le_of_not_gt hn)
          omega
        let next:=bodyExecution rest p initial current k h
        have nextInvariant:=bodyInvariant rest p initial current k h
        obtain ⟨final,ran,ready⟩ := ih (p.steps-(k+1)) (by omega) (k+1) next.val rfl (by omega) nextInvariant
        refine ⟨final,?_,ready⟩
        rw [FP.Structural.rangeInt,if_neg zero,if_pos live,next.property]
        have step : Revision001.ScaledWord.mirror p k+(p.blockInner:Int)=Revision001.ScaledWord.mirror p (k+1) := by
          simp [Revision001.ScaledWord.mirror,Nat.add_mul]
          ring
        simpa only [step] using ran
      · refine ⟨current,?_,finished k current hk h ?_⟩
        · rw [FP.Structural.rangeInt,if_neg zero,if_neg live]
        · have hx : (p.inner:Int)≤(k*p.blockInner:Int) := by
            simpa only [Revision001.ScaledWord.mirror,not_lt,sub_nonneg] using live
          exact_mod_cast hx
  exact aux _ _ _ rfl hk h

def epilogueExecution (rest : Algebra ℝ)
    (p : Parameters) (initial current : State ℝ)
    (h : OutputReady p initial current) :
    {t : State ℝ // run (backend rest) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, Revision001.ScaledWord.epilogue, original, Revision001.ScaledWord.original, typedScaled, Revision001.ScaledWord.typedScaled, VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledCandidateKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledCandidateKernel,
    ComputeKernel.surfaceBody,
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
    (h : OutputReady p initial current) :
    (epilogueExecution rest p initial current h).val.mem = (written p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputRow, outputColumn, expected, scaled, wrap,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]

theorem executionMemory (rest : Algebra ℝ)
    (p : Parameters) (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (cover : p.inner≤p.steps*p.blockInner) (initial : State ℝ) :
    ∃ final, FP.Structural.exec (backend rest) (source p) initial = some final ∧
      final.mem = (written p initial).mem := by
  let start := preludeExecution rest p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution rest p heven cover initial start.val 0 (by omega)
    (preludeInvariant rest p initial)
  let finish := epilogueExecution rest p initial mid hi
  refine ⟨finish.val, ?_, epilogueMemory rest p initial mid hi⟩
  change run (backend rest) (source p).surfaceBody initial = some finish.val
  rw [show (source p).surfaceBody = prelude p ++ [Revision001.ScaledWord.statement p] ++ epilogue p from by simpa only [source,SourcePreservation.scaled] using (show (Revision001.ScaledWord.source p).surfaceBody = prelude p ++ [Revision001.ScaledWord.statement p] ++ epilogue p from rfl),
    run_append, run_append, start.property]
  have hsingle (s : State ℝ) :
      run (backend rest) [Revision001.ScaledWord.statement p] s =
        FP.Structural.rangeInt (backend rest) "__rev_k" (Revision001.ScaledWord.mirror p 0) 0 p.blockInner (body p) s := by
    simpa [run] using Revision001.ScaledWord.statement_step (backend rest) p s
  simp only [Option.bind_some, hsingle, hm]
  exact finish.property

theorem correctness (rest : Algebra ℝ)
    (p : Parameters) (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (cover : p.inner≤p.steps*p.blockInner) (initial : State ℝ)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend rest) (source p) initial = some final ∧
      Post p initial final (expected p initial) := by
  obtain ⟨final,he,hm⟩ := executionMemory rest p heven cover initial
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
