import bench.tritonbench_g.int8_dequant_matmul.Proofs.FPExecution
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α) (divisor : α) (k : Nat) (hk : k ≤ p.steps)
    (h : Invariant M p initial divisor k current) :
    ∃ final, FP.Structural.range (backend M) "k" k p.steps 1 (body p) current = some final ∧
      Invariant M p initial divisor p.steps final := by
  have aux : ∀ n k (current : State α), p.steps - k = n → k ≤ p.steps →
      Invariant M p initial divisor k current →
      ∃ final, FP.Structural.range (backend M) "k" k p.steps 1 (body p) current = some final ∧
        Invariant M p initial divisor p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := bodyExecution M p initial current divisor k h
        have hi := bodyInvariant M p initial current divisor k h
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

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α) (divisor : α)
    (h : Invariant M p initial divisor p.steps current) :
    {t : State α // run (backend M) (epilogue p) current = some t} := by
  by_cases hno : p.hasBias = Bool.true
  · refine ⟨?_, ?_⟩
    swap
    simp [hno, epilogue, scaled, original, typed, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
      FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType,
      Region.cast, Option.bind, integralFloorDiv, integralMod,
      h.rm, h.rn, h.factor, h.wfactor, h.xfactor, h.acc, h.memory, backend]
    rfl
  · refine ⟨?_, ?_⟩
    swap
    simp [hno, epilogue, scaled, original, typed, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
      FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType,
      Region.cast, Option.bind, integralFloorDiv, integralMod,
      h.rm, h.rn, h.factor, h.wfactor, h.xfactor, h.acc, h.memory, backend]
    rfl

def written {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) (divisor : α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then
      t.write p.output (outputOffset p initial i) (.mk .fp16 (outputValue M p initial divisor i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α) (divisor : α)
    (h : Invariant M p initial divisor p.steps current) :
    (epilogueExecution M p initial current divisor h).val.mem = (written M p initial divisor).mem := by
  cases hno : p.hasBias <;>
    simp +contextual [epilogueExecution, hno, written, masked_scatter_memory, h.memory,
      active, outputOffset, outputValue, backend,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      and_assoc, and_comm, and_left_comm, Nat.mul_comm]

theorem source_decomposition (p : Parameters) :
    (source p).surfaceBody = prelude p ++ [.forRange "k" 0 p.steps 1 (body p)] ++ epilogue p := by
  rfl

def Post {α : Type} (p : Parameters) (initial final : State α) (values : CIndex p → α) : Prop :=
  (∀ i, active p initial i → final.mem p.output (outputOffset p initial i) = .mk .fp16 (values i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) (divisor : α)
    (hdiv : initial.regs .fp32 [] "divfactor" = some (fun _ => divisor))
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M) (source p) initial = some final ∧
      Post p initial final (outputValue M p initial divisor) := by
  let start := preludeExecution M p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution M p initial start.val divisor 0 (by omega)
    (preludeInvariant M p initial divisor hdiv)
  let finish := epilogueExecution M p initial mid divisor hi
  refine ⟨finish.val, ?_, ?_, ?_⟩
  · change run (backend M) (source p).surfaceBody initial = some finish.val
    rw [source_decomposition, run_append, run_append, start.property]
    have hsingle (s : State α) :
        run (backend M) [.forRange "k" 0 p.steps 1 (body p)] s =
          FP.Structural.range (backend M) "k" 0 p.steps 1 (body p) s := by
      simp [run, step]
    simp only [Option.bind_some, hsingle, hm]
    exact finish.property
  · intro i hi
    rw [epilogueMemory]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o hframe
    rw [epilogueMemory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hframe.imp id (fun h i _ => h i)) _

#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean source_decomposition
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
