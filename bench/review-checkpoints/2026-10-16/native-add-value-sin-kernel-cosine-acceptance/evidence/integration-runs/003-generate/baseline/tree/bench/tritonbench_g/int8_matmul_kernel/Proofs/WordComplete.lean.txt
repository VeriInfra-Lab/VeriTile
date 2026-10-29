import bench.tritonbench_g.int8_matmul_kernel.Proofs.WordInvariant
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem innerLoop {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (initial current : State α) (f k : Nat) (hk : k ≤ p.steps)
    (h : InnerInvariant p initial f k current) :
    ∃ final, FP.Structural.range (backend M) "j" k p.steps 1 (innerBody p) current = some final ∧
      InnerInvariant p initial f p.steps final := by
  have aux : ∀ n k (current : State α), p.steps - k = n → k ≤ p.steps →
      InnerInvariant p initial f k current →
      ∃ final, FP.Structural.range (backend M) "j" k p.steps 1 (innerBody p) current = some final ∧
        InnerInvariant p initial f p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := innerExecution M p initial current f k h
        obtain ⟨final, he, hf⟩ := ih (p.steps - (k + 1)) (by omega) (k+1) next.val rfl (by omega)
          (innerInvariant M p initial current f k h)
        refine ⟨final, ?_, hf⟩
        rw [FP.Structural.range, if_neg (by decide), if_pos hlt, next.property]
        exact he
      · have heq : k = p.steps := by omega
        subst k
        exact ⟨current, by rw [FP.Structural.range, if_neg (by decide), if_neg (by omega)], h⟩
  exact aux _ _ _ rfl hk h

theorem innerFinishesOuter {α : Type} [Inhabited α] (p : Parameters)
    (initial current : State α) (f : Nat) (h : InnerInvariant p initial f p.steps current) :
    OuterInvariant p initial (f+1) current := by
  refine ⟨h.toCommon, ?_, h.acc⟩
  simpa [aOffset, Nat.add_mul, Nat.add_assoc] using h.aptrs

theorem outerStep {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (initial current : State α) (f : Nat) (h : OuterInvariant p initial f current) :
    ∃ final, run (backend M) (outerBody p) (current.setReg "i" .nat [] (fun _ => f)) = some final ∧
      OuterInvariant p initial (f+1) final := by
  let start := outerPreludeExecution M p initial current f h
  obtain ⟨final, he, hf⟩ := innerLoop M p initial start.val f 0 (by omega)
    (outerPreludeInvariant M p initial current f h)
  refine ⟨final, ?_, innerFinishesOuter p initial final f hf⟩
  rw [outerBody, run_append, start.property]
  simpa [run, step] using he

theorem outerLoop {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (initial current : State α) (f : Nat) (hf : f ≤ 4)
    (h : OuterInvariant p initial f current) :
    ∃ final, FP.Structural.range (backend M) "i" f 4 1 (outerBody p) current = some final ∧
      OuterInvariant p initial 4 final := by
  have aux : ∀ n f (current : State α), 4-f = n → f ≤ 4 →
      OuterInvariant p initial f current →
      ∃ final, FP.Structural.range (backend M) "i" f 4 1 (outerBody p) current = some final ∧
        OuterInvariant p initial 4 final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro f current hn hf h
      by_cases hlt : f < 4
      · obtain ⟨next, he, hi⟩ := outerStep M p initial current f h
        obtain ⟨final, hrun, hfinal⟩ := ih (4-(f+1)) (by omega) (f+1) next rfl (by omega) hi
        refine ⟨final, ?_, hfinal⟩
        rw [FP.Structural.range, if_neg (by decide), if_pos hlt, he]
        exact hrun
      · have heq : f = 4 := by omega
        subst f
        exact ⟨current, by rw [FP.Structural.range, if_neg (by decide), if_neg (by omega)], h⟩
  exact aux _ _ _ rfl hf h

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (initial current : State α) (h : OuterInvariant p initial 4 current) :
    {t : State α // run (backend M) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, original, int8_matmul_kernel_surface, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod, h.pidm, h.pidn, h.acc, h.memory]
  rfl

def written {α : Type} [Inhabited α] (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then t.write p.output (outputOffset p initial i)
      (.mk .int (outerValues p initial 4 i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (initial current : State α) (h : OuterInvariant p initial 4 current) :
    (epilogueExecution M p initial current h).val.mem = (written p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]

def Post {α : Type} [Inhabited α] (p : Parameters) (initial final : State α) : Prop :=
  (∀ i, active p initial i → final.mem p.output (outputOffset p initial i) =
    .mk .int (outerValues p initial 4 i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (initial : State α)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M) (source p) initial = some final ∧ Post p initial final := by
  let start := preludeExecution M p initial
  obtain ⟨mid, hm, hi⟩ := outerLoop M p initial start.val 0 (by omega) (preludeInvariant M p initial)
  let finish := epilogueExecution M p initial mid hi
  refine ⟨finish.val, ?_, ?_, ?_⟩
  · change run (backend M) (prelude p ++ [.forRange "i" 0 4 1 (outerBody p)] ++ epilogue p) initial = some finish.val
    rw [run_append, run_append, start.property]
    have hs (s : State α) : run (backend M) [.forRange "i" 0 4 1 (outerBody p)] s =
        FP.Structural.range (backend M) "i" 0 4 1 (outerBody p) s := by simp [run, step]
    simp only [Option.bind_some, hs, hm]
    exact finish.property
  · intro i hi
    rw [epilogueMemory]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o hframe
    rw [epilogueMemory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hframe.imp id (fun h i _ => h i)) _

#axiomsClean innerLoop
#axiomsClean innerFinishesOuter
#axiomsClean outerStep
#axiomsClean outerLoop
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
