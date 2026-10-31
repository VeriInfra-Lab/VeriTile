import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordExecution
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
open VeriTile.Bench.TritonBenchG.IntScaledMatmul
open VeriTile.Triton FP.Structural FP.IntegerConversion
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance scaledCompleteDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem loopExecution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (k : Nat) (hk : k ≤ p.steps)
    (h : Invariant p initial k current) :
    ∃ final, FP.Structural.range (backend rest cfg u) "j" k p.steps 1 (body p) current = some final ∧
      Invariant p initial p.steps final := by
  have aux : ∀ n k (current : State Word), p.steps - k = n → k ≤ p.steps →
      Invariant p initial k current →
      ∃ final, FP.Structural.range (backend rest cfg u) "j" k p.steps 1 (body p) current = some final ∧
        Invariant p initial p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := bodyExecution rest cfg u p initial current k h
        have hi := bodyInvariant rest cfg u p initial current k h
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

def epilogueExecution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word)
    (h : Invariant p initial p.steps current) :
    {t : State Word // run (backend rest cfg u) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, original, typedScaled, VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.pidm, h.pidn, h.acc, h.memory, h.undefined, backend,
    FP.IntegerConversion.fp32TotalBackend, FP.IntegerConversion.fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  rfl

def written (cfg : FP.Config)
    (p : Parameters) (initial : State Word) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then
      t.write p.output (outputOffset p initial i) (.mk .int (expected cfg p initial i)) else t) initial

private theorem conversion_defined (u : Word → Int) (x : Word) (h : (toSigned 32 x).isSome) :
    totalSigned u 32 x = (toSigned 32 x).getD 0 := by
  cases hx : toSigned 32 x with
  | none => simp [hx] at h
  | some z => simp [totalSigned,hx]
attribute [local irreducible] FP.mul FP.round toSigned totalSigned

theorem epilogueMemory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word)
    (h : Invariant p initial p.steps current) (hd : InputDefined cfg p initial) :
    (epilogueExecution rest cfg u p initial current h).val.mem = (written cfg p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputRow, outputColumn, expected, scaled,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]
  apply congrArg (fun f => List.foldl f initial.mem (TileShape.allIndices [p.blockRows,p.blockColumns]))
  funext mem i
  by_cases hi : active p initial i
  · have hc := conversion_defined (u 32) _ (hd i hi)
    simp only [active,outputRow,outputColumn] at hi
    simp only [hi, ↓reduceIte]
    funext r o
    change (if r=p.output ∧ o=outputOffset p initial i then _ else mem r o) =
      (if r=p.output ∧ o=outputOffset p initial i then _ else mem r o)
    split_ifs <;> try rfl
    apply congrArg (Cell.mk .int)
    exact hc
  · simp only [active,outputRow,outputColumn] at hi
    simp only [and_comm] at hi
    simp only [hi, ↓reduceIte]


def Post (p : Parameters) (initial final : State Word) (values : CIndex p → Int) : Prop :=
  (∀ i, active p initial i → final.mem p.output (outputOffset p initial i) = .mk .int (values i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

theorem executionMemory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word) (hd : InputDefined cfg p initial) :
    ∃ final, FP.Structural.exec (backend rest cfg u) (source p) initial = some final ∧
      final.mem = (written cfg p initial).mem := by
  let start := preludeExecution rest cfg u p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution rest cfg u p initial start.val 0 (by omega)
    (preludeInvariant rest cfg u p initial)
  let finish := epilogueExecution rest cfg u p initial mid hi
  refine ⟨finish.val, ?_, epilogueMemory rest cfg u p initial mid hi hd⟩
  change run (backend rest cfg u) (source p).surfaceBody initial = some finish.val
  rw [show (source p).surfaceBody = prelude p ++ [.forRange "j" 0 p.steps 1 (body p)] ++ epilogue p from rfl,
    run_append, run_append, start.property]
  have hsingle (s : State Word) :
      run (backend rest cfg u) [.forRange "j" 0 p.steps 1 (body p)] s =
        FP.Structural.range (backend rest cfg u) "j" 0 p.steps 1 (body p) s := by simp [run,step]
  simp only [Option.bind_some, hsingle, hm]
  exact finish.property

theorem correctness (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word) (hd : InputDefined cfg p initial)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend rest cfg u) (source p) initial = some final ∧
      Post p initial final (expected cfg p initial) := by
  obtain ⟨final,he,hm⟩ := executionMemory rest cfg u p initial hd
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o hframe
    rw [hm]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hframe.imp id (fun h i _ => h i)) _

/-- The entire output memory is insensitive to arbitrary choices for
undefined conversions on masked lanes. -/
theorem undefinedPolicyIndependent (rest : Algebra Word) (cfg : FP.Config)
    (u v : Nat → Word → Int) (p : Parameters) (initial a b : State Word)
    (hd : InputDefined cfg p initial)
    (ha : FP.Structural.exec (backend rest cfg u) (source p) initial=some a)
    (hb : FP.Structural.exec (backend rest cfg v) (source p) initial=some b) : a.mem=b.mem := by
  obtain ⟨ta,hea,hma⟩ := executionMemory rest cfg u p initial hd
  obtain ⟨tb,heb,hmb⟩ := executionMemory rest cfg v p initial hd
  have hta : ta=a := Option.some.inj (hea.symm.trans ha)
  have htb : tb=b := Option.some.inj (heb.symm.trans hb)
  subst a; subst b
  exact hma.trans hmb.symm


#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean executionMemory
#axiomsClean correctness
#axiomsClean undefinedPolicyIndependent
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
