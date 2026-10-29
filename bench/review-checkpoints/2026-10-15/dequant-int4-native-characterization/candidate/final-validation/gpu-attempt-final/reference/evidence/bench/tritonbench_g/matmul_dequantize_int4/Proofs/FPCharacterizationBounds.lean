import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPCharacterizationExecution
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Characterization
open VeriTile.Triton FP.Structural FPTyped
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000

theorem a_read_extent (mode : Bool) (step : Nat) (hs : step<2) (i : AIndex (parameters mode)) :
    aOffset (parameters mode) initial step i < 4096 := by
  have hr : i.1.val<64 := i.1.isLt
  have hk : i.2.1.val<32 := i.2.1.isLt
  simp [aOffset,inputRow,inputInner,rowBlock,firstRow,rowsInGroup,blocksM,blocksN,parameters,initial]
  omega

theorem b_read_extent (mode : Bool) (step : Nat) (hs : step<2) (i : BIndex (parameters mode)) :
    bOffset (parameters mode) initial step i < 1024 := by
  have hk : i.1.val<32 := i.1.isLt
  have hn : i.2.1.val<128 := i.2.1.isLt
  simp [bOffset,inputColumn,inputInner,columnBlock,firstRow,rowsInGroup,blocksM,blocksN,parameters,initial]
  omega

theorem scale_read_extent (mode : Bool) (step : Nat) (hs : step<2)
    (j : TileIndex [(parameters mode).blockColumns]) :
    scaleOffset (parameters mode) initial step j < 256 := by
  have hn : j.1.val<128 := j.1.isLt
  cases mode <;> simp [scaleOffset,groupIndex,inputColumn,columnBlock,firstRow,rowsInGroup,
    blocksM,blocksN,parameters,initial] <;> omega

theorem zero_read_extent (mode : Bool) (step : Nat) (hs : step<2)
    (j : TileIndex [(parameters mode).blockColumns]) :
    zeroOffset (parameters mode) initial step j < 32 := by
  have hn : j.1.val<128 := j.1.isLt
  cases mode <;> simp [zeroOffset,groupIndex,inputColumn,columnBlock,firstRow,rowsInGroup,
    blocksM,blocksN,parameters,initial] <;> omega

theorem original_launch_geometry (mode : Bool) :
    (parameters mode).rows=64 ∧ (parameters mode).columns=128 ∧ (parameters mode).inner=64 ∧
    (parameters mode).blockRows=64 ∧ (parameters mode).blockColumns=128 ∧ (parameters mode).blockInner=32 ∧
    (parameters mode).groupRows=8 ∧ steps (parameters mode)=2 ∧
    (parameters mode).inner % (parameters mode).blockInner=0 ∧
    (parameters mode).columns % (parameters mode).blockColumns=0 ∧
    (parameters mode).groupSize % (parameters mode).blockInner=0 ∧
    ((parameters mode).noGroups = Bool.true ↔ (parameters mode).groupSize=(parameters mode).inner) := by
  cases mode <;> decide +kernel

#axiomsClean a_read_extent
#axiomsClean b_read_extent
#axiomsClean scale_read_extent
#axiomsClean zero_read_extent
#axiomsClean original_launch_geometry
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Characterization
