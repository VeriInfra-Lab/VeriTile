import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.BlockWordExecution
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
open VeriTile.Bench.TritonBenchG.IntScaledMatmul
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (hBK : 0 < p.blockInner) (initial current : State α) (k : Nat) (hk : k ≤ (steps p))
    (h : Invariant p initial k current) :
    ∃ final, FP.Structural.range (backend M) "k" (k * p.blockInner) p.inner p.blockInner (body p) current = some final ∧
      Invariant p initial (steps p) final := by
  have aux : ∀ n k (current : State α), (steps p) - k = n → k ≤ (steps p) →
      Invariant p initial k current →
      ∃ final, FP.Structural.range (backend M) "k" (k * p.blockInner) p.inner p.blockInner (body p) current = some final ∧
        Invariant p initial (steps p) final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < (steps p)
      · let next := bodyExecution M p initial current k h
        have hi := bodyInvariant M p initial current k h
        obtain ⟨final, he, hf⟩ := ih ((steps p) - (k + 1)) (by omega)
          (k + 1) next.val rfl (by omega) hi
        refine ⟨final, ?_, hf⟩
        have hlt' := (before_stop p hBK k).mp hlt
        rw [FP.Structural.range, if_neg (by omega), if_pos hlt', next.property]
        simpa only [Nat.add_mul, Nat.one_mul] using he
      · have heq : k = (steps p) := by omega
        subst k
        refine ⟨current, ?_, h⟩
        have hn : ¬ steps p * p.blockInner < p.inner := by
          rw [← before_stop p hBK]
          omega
        rw [FP.Structural.range, if_neg (by omega), if_neg hn]
  exact aux _ _ _ rfl hk h

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α)
    (h : Invariant p initial (steps p) current) :
    {t : State α // run (backend M) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, original, int_scaled_matmul_matmul_surface, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.pidm, h.pidn, h.acc, h.memory, backend]
  rfl

def written {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then
      t.write p.output (outputOffset p initial i) (.mk .int (outputValue p initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α)
    (h : Invariant p initial (steps p) current) :
    (epilogueExecution M p initial current h).val.mem = (written M p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputValue, backend, BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets,
    TileShape.indexToList,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm, Nat.mul_comm]

theorem source_decomposition (p : Parameters) :
    (source p).surfaceBody = prelude p ++ [.forRange "k" 0 p.inner p.blockInner (body p)] ++ epilogue p := by
  rfl

def Post {α : Type} (p : Parameters) (initial final : State α) (values : CIndex p → Int) : Prop :=
  (∀ i, active p initial i → final.mem p.output (outputOffset p initial i) = .mk .int (values i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (hBK : 0 < p.blockInner) (initial : State α)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M) (source p) initial = some final ∧
      Post p initial final (outputValue p initial) := by
  let start := preludeExecution M p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution M p hBK initial start.val 0 (by omega)
    (preludeInvariant M p initial)
  simp only [Nat.zero_mul] at hm
  let finish := epilogueExecution M p initial mid hi
  refine ⟨finish.val, ?_, ?_, ?_⟩
  · change run (backend M) (source p).surfaceBody initial = some finish.val
    rw [source_decomposition, run_append, run_append, start.property]
    have hsingle (s : State α) :
        run (backend M) [.forRange "k" 0 p.inner p.blockInner (body p)] s =
          FP.Structural.range (backend M) "k" 0 p.inner p.blockInner (body p) s := by
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
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
