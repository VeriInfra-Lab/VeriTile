import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Mathematical_FPCompiledExecution
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled
open VeriTile.Triton FP.Structural FPTyped
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α) (k : Nat) (hk : k ≤ (steps p))
    (h : Invariant M D p initial k current) :
    ∃ final, FP.Structural.range (backend M D) "k" k (steps p) 1 (body p) current = some final ∧
      Invariant M D p initial (steps p) final := by
  have aux : ∀ n k (current : State α), (steps p) - k = n → k ≤ (steps p) →
      Invariant M D p initial k current →
      ∃ final, FP.Structural.range (backend M D) "k" k (steps p) 1 (body p) current = some final ∧
        Invariant M D p initial (steps p) final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < (steps p)
      · let next := bodyExecution M D p initial current k h
        have hi := bodyInvariant M D p initial current k h
        obtain ⟨final, he, hf⟩ := ih ((steps p) - (k + 1)) (by omega)
          (k + 1) next.val rfl (by omega) hi
        refine ⟨final, ?_, hf⟩
        rw [FP.Structural.range, if_neg (by decide), if_pos hlt, next.property]
        exact he
      · have heq : k = (steps p) := by omega
        subst k
        refine ⟨current, ?_, h⟩
        rw [FP.Structural.range, if_neg (by decide), if_neg (by omega)]
  exact aux _ _ _ rfl hk h

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α)
    (h : Invariant M D p initial (steps p) current) :
    {t : State α // run (backend M D) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, source, typed, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.pidm, h.pidn, h.acc, h.memory, backend]
  rfl

def written {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then
      t.write p.output (outputOffset p initial i) (.mk .fp16 (outputValue M D p initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α)
    (h : Invariant M D p initial (steps p) current) :
    (epilogueExecution M D p initial current h).val.mem = (written M D p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputValue, backend,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]

theorem source_decomposition (p : Parameters) :
    (source p).surfaceBody = prelude p ++ [.forRangeDyn "k" (.constNat 0) (.ref .nat [] "num_pid_k") (.constNat 1) (body p)] ++ epilogue p := by
  rfl

def Post {α : Type} (p : Parameters) (initial final : State α) (values : CIndex p → α) : Prop :=
  (∀ i, active p initial i → final.mem p.output (outputOffset p initial i) = .mk .fp16 (values i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
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
    have hsingle (s : State α) (hn : s.regs .nat [] "num_pid_k" = some (fun _ => steps p)) :
        run (backend M D) [.forRangeDyn "k" (.constNat 0) (.ref .nat [] "num_pid_k") (.constNat 1) (body p)] s =
          FP.Structural.range (backend M D) "k" 0 (steps p) 1 (body p) s := by
      simp [run, step, evalOp_unfold, hn]
    simp only [Option.bind_some]
    rw [hsingle start.val (preludeInvariant M D p initial).numk, hm]
    simp only [Option.bind_some]
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
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled
