import bench.tritonbench_g.parallel_attention.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (sequentialFP32)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000

/-- A nonempty software-semantics instance with both loop segments active.
Unit tiles exercise the interpreter; this is not a claim that CUDA accepts
unit dot tiles. Native launch constraints are a separate binding obligation. -/
def roundingConfig : ForwardConfig where
  q := "q"
  k := "k"
  v := "v"
  o := "out"
  z := "z"
  s_qk_h := 2
  s_qk_t := 1
  s_qk_d := 1
  s_vo_h := 2
  s_vo_t := 1
  s_vo_d := 1
  scale := 1
  B := 1
  H := 1
  T := 2
  K := 1
  V := 1
  BTL := 1
  BTS := 1
  BK := 1
  BV := 1

def quarterUlp32 : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

def roundingInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun R off => .mk .real (if R="v" ∧ off=1 then quarterUlp32 else one32)
  regs := fun _ _ _ => none
  pids := fun d => if d=1 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def zeroIndex : TileIndex [1,1] := ⟨⟨0,by decide⟩,⟨0,by decide⟩,()⟩

theorem rounding_output [Inhabited (FP.Value .fp32)] :
    (roundedForwardValues .ieeeValue roundingConfig roundingInitial).output zeroIndex = one32 := by
  simp [roundedForwardValues, foldRange, roundedForwardUpdate, roundedQuery, sequentialDot,
    wordMatrix, roundingConfig, roundingInitial, queryChunk, keyChunk, valueChunk,
    List.finRange_succ, zeroIndex, Cell.read, one32, quarterUlp32]
  decide +kernel

specification forward_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (withDot (sequentialFP32 rest .ieeeValue) (softwareDot .ieeeValue))
      (forwardKernel roundingConfig) roundingInitial = some final ∧
      final.mem "out" 1 = .mk .real one32 ∧
      final.mem "out" 0 = roundingInitial.mem "out" 0 ∧
      (1 + 1/33554432 : Rat) ≠ 1 := by
  obtain ⟨final,he,hm⟩ := forward_software_fp32 rest .ieeeValue roundingConfig roundingInitial rfl
  refine ⟨final,he,?_,?_,by norm_num⟩
  · have hv := congrArg (fun mem => mem "out" 1) hm
    have hv' : final.mem "out" 1 = .mk .real
        ((roundedForwardValues .ieeeValue roundingConfig roundingInitial).output zeroIndex) := by
      simpa [roundedOutputMemory, roundingConfig, roundingInitial, queryChunk, valueChunk,
        TileShape.allIndices, List.finRange_succ, Fin.eq_zero, State.write, zeroIndex] using hv
    simpa only [rounding_output] using hv'
  · have hv := congrArg (fun mem => mem "out" 0) hm
    simpa [roundedOutputMemory, roundingConfig, roundingInitial, queryChunk, valueChunk,
      TileShape.allIndices, List.finRange_succ, Fin.eq_zero, State.write] using hv

#axiomsClean rounding_output
#axiomsClean forward_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
