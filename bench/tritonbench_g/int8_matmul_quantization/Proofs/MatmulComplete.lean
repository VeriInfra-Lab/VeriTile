import bench.tritonbench_g.int8_matmul_quantization.Proofs.MatmulExecution
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance matmulCompleteDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem loopExecution (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (initial current : State Word) (k : Nat) (hk : k ≤ p.steps)
    (h : Invariant p initial k current) :
    ∃ final, FP.Structural.range (backend rest cfg) "k" k p.steps 1 (body p) current = some final ∧
      Invariant p initial p.steps final := by
  have aux : ∀ n k (current : State Word), p.steps - k = n → k ≤ p.steps →
      Invariant p initial k current →
      ∃ final, FP.Structural.range (backend rest cfg) "k" k p.steps 1 (body p) current = some final ∧
        Invariant p initial p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := bodyExecution rest cfg p initial current k h
        have hi := bodyInvariant rest cfg p initial current k h
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

def epilogueExecution (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (initial current : State Word)
    (h : Invariant p initial p.steps current) :
    {t : State Word // run (backend rest cfg) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, source, typed, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.pidm, h.pidn, h.acc, h.ascale, h.bscale, h.memory, backend,
    QuantizeKvTransform.MixedBits.backend, QuantizeKvCopy.TypedQuantize.backend,
    FP.IntegerConversion.fp32TotalBackend, FP.IntegerConversion.fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  rfl

def written (cfg : FP.Config)
    (p : Parameters) (initial : State Word) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then
      t.write p.output (outputOffset p initial i) (.mk .fp16 (outputValue cfg p initial i)) else t) initial

theorem epilogueMemory (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (initial current : State Word)
    (h : Invariant p initial p.steps current) :
    (epilogueExecution rest cfg p initial current h).val.mem = (written cfg p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputValue, backend,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]


def Post (p : Parameters) (initial final : State Word) (values : CIndex p → Word) : Prop :=
  (∀ i, active p initial i → final.mem p.output (outputOffset p initial i) = .mk .fp16 (values i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

theorem correctness (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (initial : State Word)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend rest cfg) (source p) initial = some final ∧
      Post p initial final (outputValue cfg p initial) := by
  let start := preludeExecution rest cfg p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution rest cfg p initial start.val 0 (by omega)
    (preludeInvariant rest cfg p initial)
  let finish := epilogueExecution rest cfg p initial mid hi
  refine ⟨finish.val, ?_, ?_, ?_⟩
  · change run (backend rest cfg) (source p).surfaceBody initial = some finish.val
    rw [source_decomposition, run_append, run_append, start.property]
    have hsingle (s : State Word) :
        run (backend rest cfg) [.forRange "k" 0 p.steps 1 (body p)] s =
          FP.Structural.range (backend rest cfg) "k" 0 p.steps 1 (body p) s := by
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
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
