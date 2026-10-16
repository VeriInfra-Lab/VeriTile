import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeSecondPass
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
open KcacheCopyTriton.TypedCopy (scatter_memory)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
local instance completeWordDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem secondLoop (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (k : Nat) (hk : k ≤ p.steps)
    (hsep : p.input ≠ p.output) (hd : InputDefined cfg p initial)
    (h : SecondInvariant cfg p initial k current) :
    ∃ final, FP.Structural.range (backend rest cfg u) "k" k p.steps 1 (secondBody p) current = some final ∧
      SecondInvariant cfg p initial p.steps final := by
  have aux : ∀ n k (current : State Word), p.steps-k=n → k ≤ p.steps →
      SecondInvariant cfg p initial k current →
      ∃ final, FP.Structural.range (backend rest cfg u) "k" k p.steps 1 (secondBody p) current = some final ∧
        SecondInvariant cfg p initial p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := secondExecution rest cfg u p initial current k hsep h
        have hi := secondInvariant rest cfg u p initial current k hlt hsep hd h
        obtain ⟨final, he, hf⟩ := ih (p.steps-(k+1)) (by omega) (k+1) next.val rfl (by omega) hi
        refine ⟨final, ?_, hf⟩
        rw [FP.Structural.range, if_neg (by decide), if_pos hlt, next.property]
        exact he
      · have heq : k=p.steps := by omega
        subst k
        refine ⟨current, ?_, h⟩
        rw [FP.Structural.range, if_neg (by decide), if_neg (by omega)]
  exact aux _ _ _ rfl hk h

def epilogueExecution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (h : SecondInvariant cfg p initial p.steps current) :
    {t : State Word // run (backend rest cfg u) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, backend, QuantizeKvTransform.MixedBits.backend,
    QuantizeKvCopy.TypedQuantize.backend, fp32TotalBackend, fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context,
    h.pidm, h.scaleval]
  rfl

def finalWritten (cfg : FP.Config) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.blockRows]).foldl (fun t i =>
    t.write p.scales (scaleOffset p s i.1)
      (.mk .fp16 (QuantizeKvTransform.MixedBits.narrow cfg (scale cfg p s i.1))))
    (written cfg p s p.steps)

theorem epilogueMemory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (h : SecondInvariant cfg p initial p.steps current) :
    (epilogueExecution rest cfg u p initial current h).val.mem = (finalWritten cfg p initial).mem := by
  simp [epilogueExecution, finalWritten, scatter_memory, h.memory, scaleOffset,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]

theorem finalValues (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hregions : p.output ≠ p.scales)
    (hinj : ∀ a b : WriteIndex p, a.1 < p.steps → b.1 < p.steps →
      active p a.1 a.2 → active p b.1 b.2 →
      outputOffset p s a.1 a.2 = outputOffset p s b.1 b.2 → a=b)
    (i : WriteIndex p) (hi : i.1<p.steps) (ha : active p i.1 i.2) :
    (finalWritten cfg p s).mem p.output (outputOffset p s i.1 i.2) =
      .mk .int (expected cfg p s i.1 i.2) := by
  unfold finalWritten
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inl hregions)]
  exact written_readback cfg p s i hi ha hinj

theorem finalScale (cfg : FP.Config) (p : Parameters) (s : State Word) (i : Fin p.blockRows) :
    (finalWritten cfg p s).mem p.scales (scaleOffset p s i) =
      .mk .fp16 (QuantizeKvTransform.MixedBits.narrow cfg (scale cfg p s i)) := by
  apply State.scatter_readback p.scales
    (fun i : TileIndex [p.blockRows] => scaleOffset p s i.1)
    (fun i => .mk .fp16 (QuantizeKvTransform.MixedBits.narrow cfg (scale cfg p s i.1)))
    (written cfg p s p.steps) (i,PUnit.unit)
  intro a b he
  apply Prod.ext
  · apply Fin.ext
    simpa [scaleOffset] using he
  · exact Subsingleton.elim _ _

theorem finalFrame (cfg : FP.Config) (p : Parameters) (s : State Word) (r : RegionName) (o : Nat)
    (hv : r ≠ p.output ∨ ∀ k, k<p.steps → ∀ i, active p k i → o ≠ outputOffset p s k i)
    (hs : r ≠ p.scales ∨ ∀ i, o ≠ scaleOffset p s i) :
    (finalWritten cfg p s).mem r o = s.mem r o := by
  unfold finalWritten
  rw [State.scatter_frame _ _ _ _ _ _ (hs.imp id (fun h i _ => h i.1))]
  exact written_frame cfg p s p.steps r o hv

theorem expectedSpec (cfg : FP.Config) (p : Parameters) (s : State Word)
    (h : InputDefined cfg p s) (k : Nat) (hk : k<p.steps) (i : Index p) (hi : active p k i) :
    ∃ q, (quotient cfg p s k i).toRat? = some q ∧
      TowardZero q (expected cfg p s k i) ∧ SignedRange 8 (expected cfg p s k i) := by
  have hd := h k hk i hi
  cases hx : toSigned 8 (quotient cfg p s k i) with
  | none => simp [hx] at hd
  | some z => simpa [expected, hx] using (toSigned_spec 8 (quotient cfg p s k i) z).mp hx

/-- Entire two-pass source execution, not a supplied scale or a single-block
specialization. The memory result is determined by initial input bits alone. -/
theorem executionMemory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word) (hw : 0<p.blockColumns)
    (hsep : p.input ≠ p.output) (hd : InputDefined cfg p initial) :
    ∃ final, FP.Structural.exec (backend rest cfg u) (source p) initial = some final ∧
      final.mem = (finalWritten cfg p initial).mem := by
  let start := preludeExecution rest cfg u p initial
  obtain ⟨first, hfirst, hi⟩ := firstLoop rest cfg u p initial start.val 0 (by omega) hw
    (preludeInvariant rest cfg u p initial)
  let mid := middleExecution rest cfg u p initial first hi
  obtain ⟨second, hsecond, hs⟩ := secondLoop rest cfg u p initial mid.val 0 (by omega) hsep hd
    (middleInvariant rest cfg u p initial first hi)
  let finish := epilogueExecution rest cfg u p initial second hs
  refine ⟨finish.val, ?_, epilogueMemory rest cfg u p initial second hs⟩
  change run (backend rest cfg u) (source p).surfaceBody initial = some finish.val
  rw [source_decomposition, run_append, run_append, run_append, run_append, start.property]
  have hsingle (body : List ComputeStmt) (s : State Word) :
      run (backend rest cfg u) [.forRange "k" 0 p.steps 1 body] s =
        FP.Structural.range (backend rest cfg u) "k" 0 p.steps 1 body s := by simp [run, step]
  simp only [Option.bind_some, hsingle, hfirst, mid.property, hsecond]
  exact finish.property

#axiomsClean secondLoop
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean finalValues
#axiomsClean finalScale
#axiomsClean finalFrame
#axiomsClean expectedSpec
#axiomsClean executionMemory
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
