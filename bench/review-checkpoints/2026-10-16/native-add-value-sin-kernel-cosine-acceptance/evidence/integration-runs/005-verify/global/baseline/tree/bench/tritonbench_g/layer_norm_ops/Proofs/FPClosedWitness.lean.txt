import bench.tritonbench_g.layer_norm_ops.SoftwareCorrect
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPClosedWitness
open VeriTile.Triton FP.Structural FPStages FPClosed
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 1000000

def parameters : Parameters :=
  ⟨"X","Xhat","W","DY","DX","DW","DB","Mean","Rstd","C1","C2","DXBase","Dres","DresIn",
    2,2,2,2,2,2,1,2⟩
def initial : State Word where
  mem := fun r _ => .mk .fp32
    (if r=("W" : RegionName) then FP.Value.ofBits .fp32 0x3f800000
      else if r=("DY" : RegionName) then FP.Value.ofBits .fp32 0x40000000
      else FP.Value.zero .fp32)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => match d with
    | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => FP.Value.infinity .fp32 Bool.false
    | d => defaultValue d

/-- W's masked tail is undefined, whereas DY's masked tail is +0. Every
BLOCK_N product is still evaluated. Inf*0 must reach the sum as NaN. -/
theorem tail_product_is_retained : bitC2 parameters initial = FP.Value.qNaN .fp32 := by
  decide +kernel

theorem nonempty_c2_source_success :
    ∃ final, FP.Structural.exec (backend parameters.columns) (c2Kernel parameters) initial=some final ∧
      final.mem "C2" 0=.mk .fp32 (FP.Value.qNaN .fp32) ∧
      final.mem "DY" 0=initial.mem "DY" 0 := by
  obtain ⟨final,he,hv,hf⟩ := SoftwareCorrect.software_fp32_c2_reduction_correctness parameters initial
  refine ⟨final,he,?_,hf "DY" 0 (Or.inl (by decide))⟩
  rw [tail_product_is_retained] at hv
  exact hv

/-- +0 accumulator adds are retained, including the signed-zero result. -/
theorem positive_zero_add_is_not_identity :
    FP.add .ieeeValue (FP.Value.zero .fp32) (FP.Value.zero .fp32 Bool.true) = FP.Value.zero .fp32 ∧
    FP.add .ieeeValue (FP.Value.zero .fp32) (FP.Value.zero .fp32 Bool.true) ≠ FP.Value.zero .fp32 Bool.true := by
  decide +kernel

#axiomsClean tail_product_is_retained
#axiomsClean nonempty_c2_source_success
#axiomsClean positive_zero_add_is_not_identity
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPClosedWitness
