import bench.tritonbench_g.iv_dependent_matmul.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000

def witnessParameters : Parameters := ⟨"X","W","C",3,3,4,6,1,1,6,5,1,2,2,2⟩
def witnessInitial : State Word where
  mem := fun r o => .mk .fp32 (FP.Value.ofBits .fp32
    (if r=("X":RegionName) then
      if o%6=0 then 0x3f800000 else if o%6=1 then 0x40000000 else if o%6=2 then 0x40400000 else 0x40800000
     else if r=("W":RegionName) then 0x40000000 else 0x42c60000))
  regs := fun _ _ _ => none
  pids := fun _ => 3
  numPids := fun _ => 4
  undef := fun d _ _ => defaultValue d

def lane00 : TileIndex [2,2] := (⟨0,by decide⟩,⟨0,by decide⟩,())
theorem witness_domain : 0<witnessParameters.blockInner ∧ witnessParameters.inner=witnessParameters.blockInner*2 := by
  decide +kernel

theorem witness_value (g : Graph) :
    outputValues .ieeeValue witnessParameters
      (accumulated .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue) g witnessParameters witnessInitial)
      lane00=FP.Value.ofBits .fp32 0x41a00000 := by
  cases g <;> decide +kernel

theorem witness_layout : Function.Injective (outputOffset witnessParameters witnessInitial) :=
  output_injective witnessParameters witnessInitial rfl (by decide)

theorem two_block_bit_success (rest : Algebra Word) (g : Graph) :
    ∃ final, FP.Structural.exec (backend rest .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue))
      (source g witnessParameters) witnessInitial=some final ∧
      final.mem "C" 12=.mk .fp32 (FP.Value.ofBits .fp32 0x41a00000) ∧
      final.mem "C" 13=witnessInitial.mem "C" 13 ∧
      final.mem "C" 17=witnessInitial.mem "C" 17 ∧
      final.mem "X" 12=witnessInitial.mem "X" 12 := by
  obtain ⟨final,he,hp⟩ := correctness rest .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue)
    g witnessParameters witnessInitial witness_layout
  refine ⟨final,he,?_,?_,?_,hp.2 "X" 12 (Or.inl (by decide))⟩
  · have ha : active witnessParameters witnessInitial lane00=Bool.true := by
      simp [active,row,col,rowBlock,colBlock,columnsGrid,witnessParameters,witnessInitial,lane00]
    have hv := hp.1 lane00 ha
    change final.mem "C" 12=_ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) (witness_value g))
  · apply hp.2 "C" 13
    right
    intro i hi
    simp [active,row,col,rowBlock,colBlock,columnsGrid,witnessParameters,witnessInitial] at hi
    have hm : i.1.val=0 := by omega
    have hn : i.2.1.val=0 := by omega
    simp [outputOffset,row,col,rowBlock,colBlock,columnsGrid,witnessParameters,witnessInitial,hm,hn]
  · apply hp.2 "C" 17
    right
    intro i hi
    simp [active,row,col,rowBlock,colBlock,columnsGrid,witnessParameters,witnessInitial] at hi
    have hm : i.1.val=0 := by omega
    have hn : i.2.1.val=0 := by omega
    simp [outputOffset,row,col,rowBlock,colBlock,columnsGrid,witnessParameters,witnessInitial,hm,hn]

/-- A fp32 half-way value loses its low mantissa bits at the intermediate half
cast. The widening store must not bypass this narrowing. -/
theorem intermediate_half_rounding :
    castWord .ieeeValue .half .single
      (castWord .ieeeValue .single .half (FP.Value.ofBits .fp32 0x3f801000))=
      FP.Value.ofBits .fp32 0x3f800000 ∧
    castWord .ieeeValue .half .single
      (castWord .ieeeValue .single .half (FP.Value.ofBits .fp32 0x80000000))=
      FP.Value.ofBits .fp32 0x80000000 := by decide +kernel
#axiomsClean witness_domain
#axiomsClean witness_value
#axiomsClean witness_layout
#axiomsClean two_block_bit_success
#axiomsClean intermediate_half_rounding
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
