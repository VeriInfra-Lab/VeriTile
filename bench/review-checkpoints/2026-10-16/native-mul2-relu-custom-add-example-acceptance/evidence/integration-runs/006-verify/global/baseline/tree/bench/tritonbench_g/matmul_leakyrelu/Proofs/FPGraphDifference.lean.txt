import bench.tritonbench_g.matmul_leakyrelu.FPCorrect
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.GraphDifference
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000

def sourceParameters : FPValue.Parameters where
  A := "A"; B := "B"; C := "C"
  M := 1; N := 1; K := 6
  stride_am := 6; stride_ak := 1; stride_bk := 1; stride_bn := 1
  stride_cm := 1; stride_cn := 1
  BLOCK_SIZE_M := 1; BLOCK_SIZE_N := 1; BLOCK_SIZE_K := 2
  GROUP_SIZE_M := 1; ACTIVATION := Bool.false

def candidateParameters : FPFused.Parameters where
  A := "A"; B := "B"; C := "C"
  M := 1; N := 1; K := 6
  stride_am := 6; stride_ak := 1; stride_bk := 1; stride_bn := 1
  stride_cm := 1; stride_cn := 1
  BLOCK_SIZE_M := 1; BLOCK_SIZE_N := 1; BLOCK_SIZE_K := 2
  GROUP_SIZE_M := 1; ACTIVATION := Bool.false

/-- Half inputs form blocks [4096*4096,0], [1,-1], [-4096*4096,0].
The two graphs differ after the middle block; cancellation in the last block
exposes that difference in the final half output, without overflow or NaNs. -/
def initial : State Word where
  mem := fun r o => .mk .fp16 (FP.Value.ofBits .fp32
    (if r=("A" : RegionName) then
      if o=0 then 0x6c00 else if o=2 then 0x3c00 else if o=3 then 0xbc00 else if o=4 then 0xec00 else 0
    else if r=("B" : RegionName) then if o=0 ∨ o=4 then 0x6c00 else 0x3c00
    else 0x4000))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane : TileIndex [1,1] := (⟨0,by decide⟩,⟨0,by decide⟩,())

theorem distinct_final_bit_values :
    FPValue.bitOutput sourceParameters initial lane=FP.Value.ofBits .fp32 0 ∧
    FPFused.bitOutput candidateParameters initial lane=FP.Value.ofBits .fp32 0xbc00 := by
  decide +kernel

/-- Both independent contracts execute successfully on the same initial
storage, with genuinely different final half words. This refutes an
unconditional equality between these software profiles; it assigns neither
profile to hardware without a separately checked native binding. -/
theorem two_graphs_execute_with_different_outputs (rest : Algebra Word) :
    ∃ sourceFinal candidateFinal,
      FP.Structural.exec (FPValue.backend (FPValue.software rest) FPValue.sequentialDot)
        (FPValue.source sourceParameters .half) initial=some sourceFinal ∧
      FP.Structural.exec (FPFused.backend (FPFused.software rest) FPFused.sequentialDotAcc)
        (FPFused.source candidateParameters .half) initial=some candidateFinal ∧
      sourceFinal.mem "C" 0=Cell.mk .fp16 (FP.Value.ofBits .fp32 0) ∧
      candidateFinal.mem "C" 0=Cell.mk .fp16 (FP.Value.ofBits .fp32 0xbc00) ∧
      sourceFinal.mem "C" 0≠candidateFinal.mem "C" 0 := by
  obtain ⟨s,hs,vs,fs⟩ := FPCorrect.half_software_correctness rest sourceParameters initial rfl (by decide)
  obtain ⟨c,hc,vc,fc⟩ := FPCorrect.fused_half_software_correctness rest candidateParameters initial rfl (by decide)
  have hvs := vs lane (by decide)
  have hvc := vc lane (by decide)
  rw [distinct_final_bit_values.1] at hvs
  rw [distinct_final_bit_values.2] at hvc
  change s.mem "C" 0=Cell.mk .fp16 (FP.Value.ofBits .fp32 0) at hvs
  change c.mem "C" 0=Cell.mk .fp16 (FP.Value.ofBits .fp32 0xbc00) at hvc
  refine ⟨s,c,hs,hc,hvs,hvc,?_⟩
  rw [hvs,hvc]
  intro h
  have hh := congrArg (fun cell : Cell Word => cell.read .fp16) h
  have hn : FP.Value.ofBits .fp32 0 ≠ FP.Value.ofBits .fp32 0xbc00 := by decide
  exact hn hh

#axiomsClean distinct_final_bit_values
#axiomsClean two_graphs_execute_with_different_outputs
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.GraphDifference
