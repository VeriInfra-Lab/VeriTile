import bench.tritonbench_g.matmul_triton_autotune.Proofs.Real
import bench.tritonbench_g.matmul_triton_autotune.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
open VeriTile.Triton FP.Structural
noncomputable section

/-- The selected public precision and all three exact source configurations. -/
theorem configuration_witness :
    configurations "cuda"=[.cuda128,.cuda64] ∧ configurations "hip"=[.hip128] ∧
    ((Config.cuda128.bm,Config.cuda128.bn,Config.cuda128.bk,Config.cuda128.gm),
      Config.cuda128.stages,Config.cuda128.warps,Config.cuda128.waves)=((128,256,64,8),3,8,none) ∧
    ((Config.cuda64.bm,Config.cuda64.bn,Config.cuda64.bk,Config.cuda64.gm),
      Config.cuda64.stages,Config.cuda64.warps,Config.cuda64.waves)=((64,256,32,8),4,4,none) ∧
    ((Config.hip128.bm,Config.hip128.bn,Config.hip128.bk,Config.hip128.gm),
      Config.hip128.stages,Config.hip128.warps,Config.hip128.waves)=((128,256,16,1),2,4,some 2) := by decide

/-- Narrow output, partial final M group, ragged K, and transposed B strides. -/
def raggedRequest (c : Config) (enabled : String) : Request where
  A := "A"; B := "B"; C := "C"
  rows := 1025; inner := 65; innerB := 65; columns := 1
  strideAM := 65; strideAK := 1; strideBK := 1; strideBN := 65
  aContiguous := Bool.true; activation := enabled
  target := if c=.hip128 then "hip" else "cuda"
  config := c

theorem trips_witness :
    (raggedRequest .cuda128 "").parameters.numKBlocks=2 ∧
    (raggedRequest .cuda64 "").parameters.numKBlocks=3 ∧
    (raggedRequest .hip128 "").parameters.numKBlocks=5 ∧
    (raggedRequest .cuda128 "leaky_relu").parameters.ACTIVATION=Bool.true ∧
    (raggedRequest .cuda128 "Leaky_ReLU").parameters.ACTIVATION=Bool.false := by decide

theorem narrow_injection {α : Type} (c : Config) (s : State α)
    (i j : FPValue.CIndex (raggedRequest c "").parameters)
    (hi : FPValue.active (raggedRequest c "").parameters s i)
    (hj : FPValue.active (raggedRequest c "").parameters s j)
    (he : FPValue.cOffset (raggedRequest c "").parameters s i=
      FPValue.cOffset (raggedRequest c "").parameters s j) : i=j :=
  active_injective (raggedRequest c "") s i j hi hj he

#axiomsClean configuration_witness
#axiomsClean trips_witness
#axiomsClean narrow_injection
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
