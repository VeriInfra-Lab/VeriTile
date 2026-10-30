import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SignedSetup
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.Direct.FPValues

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPSetup
open VeriTile.Triton FP.Structural FPProgram FPValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Constructive evaluation of all three actual metadata-prefix statements. -/
def preguard (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row : Nat) (data : Data α) : {s : State α //
    run (machine M reduction) (prelude h) (entry h batch row data)=some s} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude,code,primarySource,Kernels.original,
    VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.original,
    VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.specialize,
    ComputeKernel.surfaceBody,run,step,FP.Structural.evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,entry,Data.cells,State.setReg,Cell.read,Region.cast]
  rfl

/-- Every setup statement succeeds for the local primitives, including the
source's explicit fp32-to-half cast and signed metadata-derived pointers. -/
def preloop (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) :
    {s : State α // run (machine M reduction) (setup h)
      (preguard M reduction h batch row data).val=some s} := by
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  cases ev
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [setup,active,code,primarySource,Kernels.original,
      VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.original,
      VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.specialize,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,
      Option.map_some,Option.getD_some,beq_self_eq_true]
    dsimp [preguard]
    simp (config := {maxSteps := 100000}) [Source.dynamicRange?,Option.map,Option.getD,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,Cell.read,ofFloat,toFloat,Region.cast,entry,Data.cells,metadata]
    try rfl

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPSetup
