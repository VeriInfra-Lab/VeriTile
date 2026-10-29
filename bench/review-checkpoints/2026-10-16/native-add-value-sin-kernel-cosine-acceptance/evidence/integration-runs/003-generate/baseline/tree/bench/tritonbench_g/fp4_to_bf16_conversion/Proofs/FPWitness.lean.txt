import bench.tritonbench_g.fp4_to_bf16_conversion.FPCorrect
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def witnessParameters : Parameters := ⟨"bytes","scales","bytes",17,8⟩
def witnessState (scale undefNat : Nat) : State Word :=
  { mem := fun r _ => .mk .nat (if r="scales" then scale else 34)
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 1
    undef := fun d _ _ => match d with
      | .nat => undefNat
      | d => defaultValue d }

def witnessLane (n : Nat) (h : n<256) : TileIndex [witnessParameters.block*2] :=
  (⟨n,h⟩,PUnit.unit)

theorem witness_typed (scale u : Nat) : ActiveBytes witnessParameters (witnessState scale u) := by
  intro i hi
  exact ⟨34,rfl⟩

theorem witness_values :
    expectedValue software witnessParameters (witnessState 127 127) (witnessLane 0 (by decide)) = embed ⟨0x3f80⟩ ∧
    expectedValue software witnessParameters (witnessState 127 127) (witnessLane 32 (by decide)) = embed ⟨0x3f80⟩ ∧
    expectedValue software witnessParameters (witnessState 127 128) (witnessLane 32 (by decide)) = embed ⟨0x4000⟩ ∧
    expectedValue software witnessParameters (witnessState 127 128) (witnessLane 33 (by decide)) = embed ⟨0x4000⟩ ∧
    expectedValue software witnessParameters (witnessState 255 127) (witnessLane 0 (by decide)) = embed ⟨0x7fc0⟩ := by
  decide +kernel

/-- Actual nonempty BLOCK=128 candidate execution with aliased byte input and
bf16 output. The final active group intentionally has a masked scale load.
Both chosen undef values below are valid uint8 payloads. -/
theorem alias_tail_execution (rest : Algebra Word) (u : Nat) :
    ∃ final, FP.Structural.exec (backend rest software) (source witnessParameters)
        (witnessState 127 u) = some final ∧
      final.mem "bytes" 32 = .mk .bf16 (expectedValue software witnessParameters
        (witnessState 127 u) (witnessLane 32 (by decide))) ∧
      final.mem "bytes" 33 = .mk .bf16 (expectedValue software witnessParameters
        (witnessState 127 u) (witnessLane 33 (by decide))) ∧
      final.mem "bytes" 34 = .mk .nat 34 ∧
      final.mem "scales" 0 = .mk .nat 127 := by
  obtain ⟨final,he,hm,hv,hf⟩ := FPCorrect.fp4_to_bf16_scaled_candidate_machine_correctness
    rest witnessParameters (witnessState 127 u) (witness_typed _ _)
  refine ⟨final,he,?_,?_,?_,?_⟩
  · exact hv (witnessLane 32 (by decide)) (by
      simp [active, Fp4ToBf16.FPValue.active, Fp4ToBf16.FPValue.outputOffset, base, witnessParameters, witnessState, witnessLane])
  · exact hv (witnessLane 33 (by decide)) (by
      simp [active, Fp4ToBf16.FPValue.active, Fp4ToBf16.FPValue.outputOffset, base, witnessParameters, witnessState, witnessLane])
  · have h := hf "bytes" 34 (Or.inr (by
      intro j hj
      simp only [active, Fp4ToBf16.FPValue.active, Fp4ToBf16.FPValue.outputOffset,
        base, witnessParameters, witnessState, Nat.zero_mul, Nat.zero_add] at hj
      simpa [outputOffset, Fp4ToBf16.FPValue.outputOffset, base, witnessParameters, witnessState] using Nat.ne_of_gt hj))
    exact h
  · exact hf "scales" 0 (Or.inl (by decide))

/-- Source-level counterexample to unconditional undef independence: count17
uses floor(34/32)=1 scale, but outputs32 and33 need the masked second scale.
This is an address/mask issue, not an empirically testable numerical law. -/
theorem partial_group_undef_counterexample (rest : Algebra Word) :
    ∃ left right, FP.Structural.exec (backend rest software) (source witnessParameters)
        (witnessState 127 127) = some left ∧
      FP.Structural.exec (backend rest software) (source witnessParameters)
        (witnessState 127 128) = some right ∧
      left.mem "bytes" 32 = .mk .bf16 (embed ⟨0x3f80⟩) ∧
      right.mem "bytes" 32 = .mk .bf16 (embed ⟨0x4000⟩) ∧
      left.mem "bytes" 32 ≠ right.mem "bytes" 32 := by
  obtain ⟨left,hl,hvl,_⟩ := alias_tail_execution rest 127
  obtain ⟨right,hr,hvr,_⟩ := alias_tail_execution rest 128
  rw [witness_values.2.1] at hvl
  rw [witness_values.2.2.1] at hvr
  refine ⟨left,right,hl,hr,hvl,hvr,?_⟩
  rw [hvl,hvr]
  intro h
  have hn : (16256 : Nat) = 16384 := congrArg (fun c : Cell Word => (c.read .bf16).bits.toNat) h
  omega

/-- NaN scale handling occurs in complete source execution, after the pow128
intermediate overflows to fp32 infinity and before bf16 multiplication. -/
theorem nan_scale_execution (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest software) (source witnessParameters)
        (witnessState 255 127) = some final ∧
      final.mem "bytes" 0 = .mk .bf16 (embed ⟨0x7fc0⟩) := by
  obtain ⟨final,he,hm,hv,hf⟩ := FPCorrect.fp4_to_bf16_scaled_candidate_machine_correctness
    rest witnessParameters (witnessState 255 127) (witness_typed _ _)
  refine ⟨final,he,?_⟩
  have h := hv (witnessLane 0 (by decide)) (by decide)
  rw [witness_values.2.2.2.2] at h
  exact h

#axiomsClean witness_values
#axiomsClean alias_tail_execution
#axiomsClean partial_group_undef_counterexample
#axiomsClean nan_scale_execution
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
