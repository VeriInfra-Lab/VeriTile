import bench.tritonbench_g.matmul_triton2.FPCorrect
namespace VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
open VeriTile.Triton FP.Structural FP.Equational
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Last partial M group, nonzero pid, nonunit input strides, three K blocks,
input/output alias and both masked output dimensions. Only lane (0,0) writes:
A[30]; masked A[31] and A[37] retain their original values, even though they
are also among the original input reads. -/
def witnessParameters : Parameters :=
  ⟨"A","B","A",5,3,7,1,5,2,7,1,2,2,2,2,3⟩
def witnessInitial : State Word where
  mem := fun r _ => if r="A" ∨ r="B" then .mk .fp32 (FP.Value.ofBits .fp32 0x3f800001)
    else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then 5 else 0
  numPids := fun _ => 6
  undef := fun d _ _ => defaultValue d

def witnessExpected : InputMode → Word
  | .fullFP32 => FP.Value.ofBits .fp32 0x40c00003
  | .tf32NearestEven | .tf32NearestAway => FP.Value.ofBits .fp32 0x40c00000

theorem witness_values (mode : InputMode) : ∀ i : CIndex witnessParameters,
    expected mode .ieeeValue ReductionPlan.withSeed witnessParameters witnessInitial i=witnessExpected mode := by
  cases mode <;> decide +kernel

/-- All source operations execute for each named software precision profile.
The different full-fp32/TF32 results are supporting evidence, not a claim about
which profile the unspecified native default selects. -/
theorem witness_execution (rest : Algebra Word) (mode : InputMode) :
    ∃ final, FP.Structural.exec (backend rest (software mode .ieeeValue ReductionPlan.withSeed))
        (source witnessParameters) witnessInitial=some final ∧
      (∀ i : CIndex witnessParameters, active witnessParameters witnessInitial i →
        final.mem "A" (cOffset witnessParameters witnessInitial i)=.mk .fp32 (witnessExpected mode)) ∧
      final.mem "A" 31=witnessInitial.mem "A" 31 ∧
      final.mem "A" 37=witnessInitial.mem "A" 37 ∧
      final.mem "B" 0=witnessInitial.mem "B" 0 := by
  obtain ⟨final,he,hv,hf,_⟩ := FPCorrect.matmul_triton2_machine_correctness rest mode .ieeeValue
    ReductionPlan.withSeed witnessParameters witnessInitial (by decide)
    (rowMajor_injective _ _ (by decide) (by decide))
  refine ⟨final,he,?_,?_,?_,?_⟩
  · intro i hi
    simpa only [witness_values] using hv i hi
  · exact hf "A" 31 (Or.inr (by decide +kernel))
  · exact hf "A" 37 (Or.inr (by decide +kernel))
  · exact hf "B" 0 (Or.inl (by decide))

#axiomsClean witness_values
#axiomsClean witness_execution
end
end VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
