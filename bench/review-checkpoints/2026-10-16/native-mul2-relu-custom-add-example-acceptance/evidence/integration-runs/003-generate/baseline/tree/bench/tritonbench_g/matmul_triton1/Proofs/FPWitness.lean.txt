import bench.tritonbench_g.matmul_triton1.FPCorrect
namespace VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
open VeriTile.Triton FP.Structural FP.Equational
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Three complete K blocks and a nonzero program ID; output aliases input.
The output tile is rows 2/3, columns 2/3, with untouched row gaps. -/
def witnessParameters : Parameters := ⟨"X","Y","X",4,4,2,2,2,3⟩
def witnessInitial : State Word where
  mem := fun r _ => if r="X" ∨ r="Y" then .mk .fp32 (FP.Value.ofBits .fp32 0x3f800001)
    else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then 3 else 0
  numPids := fun _ => 4
  undef := fun d _ _ => defaultValue d

def witnessExpected : Word := FP.Value.ofBits .fp32 0x40c00003

theorem witness_values : ∀ i : CIndex witnessParameters,
    expected .ieeeValue ReductionPlan.withSeed witnessParameters witnessInitial i=witnessExpected := by
  decide +kernel

/-- A fixed regression supporting, not replacing, the parameterized contract:
all 3 K iterations execute, all 4 outputs are checked, and aliased input cells
outside the output tile retain their original bit patterns. -/
theorem witness_execution (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest (software .ieeeValue ReductionPlan.withSeed))
        (source witnessParameters) witnessInitial=some final ∧
      (∀ i : CIndex witnessParameters, final.mem "X" (cOffset witnessParameters witnessInitial i)=
        .mk .fp32 witnessExpected) ∧
      final.mem "X" 12=witnessInitial.mem "X" 12 ∧
      final.mem "Y" 0=witnessInitial.mem "Y" 0 := by
  obtain ⟨final,he,hv,hf,_⟩ := FPCorrect.matmul_triton1_machine_correctness rest .ieeeValue
    ReductionPlan.withSeed witnessParameters witnessInitial (by decide) (by decide)
  refine ⟨final,he,?_,?_,?_⟩
  · intro i
    simpa only [witness_values] using hv i
  · apply hf "X" 12
    right
    intro i
    have hr := i.1.isLt
    have hc := i.2.1.isLt
    norm_num [witnessParameters] at hr hc
    norm_num [cOffset,row,column,columnBlocks,witnessParameters,witnessInitial]
    omega
  · exact hf "Y" 0 (Or.inl (by decide))

#axiomsClean witness_values
#axiomsClean witness_execution
end
end VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
