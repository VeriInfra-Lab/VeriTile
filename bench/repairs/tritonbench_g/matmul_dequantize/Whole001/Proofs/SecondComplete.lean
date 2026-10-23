import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.SecondExecution
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPTypedSecond
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (k : Nat) (hk : k ≤ p.steps)
    (h : Invariant M D p initial k current) :
    ∃ final, FP.Structural.range (backend M D) "k" k p.steps 1 (body p) current = some final ∧
      Invariant M D p initial p.steps final := by
  have aux : ∀ n k (current : State α), p.steps - k = n → k ≤ p.steps →
      Invariant M D p initial k current →
      ∃ final, FP.Structural.range (backend M D) "k" k p.steps 1 (body p) current = some final ∧
        Invariant M D p initial p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := bodyExecution M D p initial current k h
        have hi := bodyInvariant M D p initial current k h
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

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α)
    (h : Invariant M D p initial p.steps current) :
    {t : State α // run (backend M D) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, source, FPTyped.matmulTyped, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.pidm, h.pidn, h.acc, h.memory, backend]
  rfl

def written {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then
      t.write p.output (outputOffset p initial i) (.mk .fp16 (outputValue M D p initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α)
    (h : Invariant M D p initial p.steps current) :
    (epilogueExecution M D p initial current h).val.mem = (written M D p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputValue, backend,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]

theorem source_decomposition (p : Parameters) :
    (source p).surfaceBody = prelude p ++ [.forRange "k" 0 p.steps 1 (body p)] ++ epilogue p := by
  rfl

def Post {α : Type} (p : Parameters) (initial final : State α) (values : CIndex p → α) : Prop :=
  (∀ i, active p initial i → final.mem p.output (outputOffset p initial i) = .mk .fp16 (values i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial : State α)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M D) (source p) initial = some final ∧
      Post p initial final (outputValue M D p initial) := by
  let start := preludeExecution M D p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution M D p initial start.val 0 (by omega)
    (preludeInvariant M D p initial)
  let finish := epilogueExecution M D p initial mid hi
  refine ⟨finish.val, ?_, ?_, ?_⟩
  · change run (backend M D) (source p).surfaceBody initial = some finish.val
    rw [source_decomposition, run_append, run_append, start.property]
    have hsingle (s : State α) :
        run (backend M D) [.forRange "k" 0 p.steps 1 (body p)] s =
          FP.Structural.range (backend M D) "k" 0 p.steps 1 (body p) s := by
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
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPTypedSecond
