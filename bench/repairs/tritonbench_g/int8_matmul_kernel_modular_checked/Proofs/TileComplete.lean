import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileInvariant
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileConfig
namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 (Byte)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem innerLoop (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters)
    (initial current : State Word32) (f k : Nat) (hf : f < 4) (hk : k ≤ p.steps) (defined : InputDefined p initial)
    (h : InnerInvariant cast p initial f k current) :
    ∃ final, FP.Structural.range (backend cast M) "j" k p.steps 1 (innerBody p) current = some final ∧
      InnerInvariant cast p initial f p.steps final := by
  have aux : ∀ n k (current : State Word32), p.steps - k = n → k ≤ p.steps →
      InnerInvariant cast p initial f k current →
      ∃ final, FP.Structural.range (backend cast M) "j" k p.steps 1 (innerBody p) current = some final ∧
        InnerInvariant cast p initial f p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := innerExecution cast M p initial current f k hf hlt defined h
        obtain ⟨final, he, hf⟩ := ih (p.steps - (k + 1)) (by omega) (k+1) next.val rfl (by omega)
          (innerInvariant cast M p initial current f k hf hlt defined h)
        refine ⟨final, ?_, hf⟩
        rw [FP.Structural.range, if_neg (by decide), if_pos hlt, next.property]
        exact he
      · have heq : k = p.steps := by omega
        subst k
        exact ⟨current, by rw [FP.Structural.range, if_neg (by decide), if_neg (by omega)], h⟩
  exact aux _ _ _ rfl hk h

theorem innerFinishesOuter (cast : Word32 → Byte) (p : Parameters)
    (initial current : State Word32) (f : Nat) (h : InnerInvariant cast p initial f p.steps current) :
    OuterInvariant cast p initial (f+1) current := by
  refine ⟨h.toCommon, ?_, h.acc⟩
  simpa [aOffset, Nat.add_mul, Nat.add_assoc] using h.aptrs

theorem outerStep (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters)
    (initial current : State Word32) (f : Nat) (hf : f < 4) (defined : InputDefined p initial)
    (h : OuterInvariant cast p initial f current) :
    ∃ final, run (backend cast M) (outerBody p) (current.setReg "i" .nat [] (fun _ => f)) = some final ∧
      OuterInvariant cast p initial (f+1) final := by
  let start := outerPreludeExecution cast M p initial current f h
  obtain ⟨final, he, hf⟩ := innerLoop cast M p initial start.val f 0 hf (by omega) defined
    (outerPreludeInvariant cast M p initial current f h)
  refine ⟨final, ?_, innerFinishesOuter cast p initial final f hf⟩
  rw [outer_decomposition, run_append, start.property]
  simpa [run, step, evalOp_unfold, stepsExpr, Parameters.steps, numeric, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, integralFloorDiv] using he

theorem outerLoop (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters)
    (initial current : State Word32) (f : Nat) (hf : f ≤ 4) (defined : InputDefined p initial)
    (h : OuterInvariant cast p initial f current) :
    ∃ final, FP.Structural.range (backend cast M) "i" f 4 1 (outerBody p) current = some final ∧
      OuterInvariant cast p initial 4 final := by
  have aux : ∀ n f (current : State Word32), 4-f = n → f ≤ 4 →
      OuterInvariant cast p initial f current →
      ∃ final, FP.Structural.range (backend cast M) "i" f 4 1 (outerBody p) current = some final ∧
        OuterInvariant cast p initial 4 final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro f current hn hf h
      by_cases hlt : f < 4
      · obtain ⟨next, he, hi⟩ := outerStep cast M p initial current f hlt defined h
        obtain ⟨final, hrun, hfinal⟩ := ih (4-(f+1)) (by omega) (f+1) next rfl (by omega) hi
        refine ⟨final, ?_, hfinal⟩
        rw [FP.Structural.range, if_neg (by decide), if_pos hlt, he]
        exact hrun
      · have heq : f = 4 := by omega
        subst f
        exact ⟨current, by rw [FP.Structural.range, if_neg (by decide), if_neg (by omega)], h⟩
  exact aux _ _ _ rfl hf h

def epilogueExecution (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters)
    (initial current : State Word32) (h : OuterInvariant cast p initial 4 current) :
    {t : State Word32 // run (backend cast M) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, source, compileAssertionProjection, quotedSource, kernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod, h.pidm, h.pidn, h.acc, h.memory]
  rfl

def written (cast : Word32 → Byte) (p : Parameters) (initial : State Word32) : State Word32 :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then t.write p.output (outputOffset p initial i)
      (.mk .int (outerValues cast p initial 4 i)) else t) initial

theorem epilogueMemory (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters)
    (initial current : State Word32) (h : OuterInvariant cast p initial 4 current) :
    (epilogueExecution cast M p initial current h).val.mem = (written cast p initial).mem := by
  have cast32 (i : CIndex p) :
      (if 2 * ((outerValues cast p initial 4 i % 4294967296).toNat % 4294967296) < 4294967296 then
        max (outerValues cast p initial 4 i % 4294967296) 0 % 4294967296
       else max (outerValues cast p initial 4 i % 4294967296) 0 % 4294967296 - 4294967296) =
        outerValues cast p initial 4 i := by
    have normalize (x : Int) :
        (if 2 * ((x % 4294967296).toNat % 4294967296) < 4294967296 then
          max (x % 4294967296) 0 % 4294967296
         else max (x % 4294967296) 0 % 4294967296 - 4294967296) = wrap 32 x := by
      simpa [IntegerExpr.signedBits, wrap] using signedBits_int_mod 32 x
    rw [normalize]
    obtain ⟨word,hw⟩ := outerValues_word cast p initial 4 i
    rw [hw]
    simp only [wrap,BitVec.ofInt_toInt]
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,cast32,
    active, outputOffset, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]

def Post (cast : Word32 → Byte) (p : Parameters) (initial final : State Word32) : Prop :=
  (∀ i, active p initial i → final.mem p.output (outputOffset p initial i) =
    .mk .int (outerValues cast p initial 4 i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

theorem correctness (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters) (initial : State Word32)
    (_admitted : StaticAdmission p) (defined : InputDefined p initial)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend cast M) (source p) initial = some final ∧ Post cast p initial final := by
  let start := preludeExecution cast M p initial
  obtain ⟨mid, hm, hi⟩ := outerLoop cast M p initial start.val 0 (by omega) defined (preludeInvariant cast M p initial)
  let finish := epilogueExecution cast M p initial mid hi
  refine ⟨finish.val, ?_, ?_, ?_⟩
  · change run (backend cast M) (source p).surfaceBody initial = some finish.val
    rw [source_decomposition]
    rw [run_append, run_append, start.property]
    have hs (s : State Word32) : run (backend cast M) [.forLoop "i" 4 (outerBody p)] s =
        FP.Structural.range (backend cast M) "i" 0 4 1 (outerBody p) s := by simp [run, step, evalOp_unfold, loop_eq_range]
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
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
