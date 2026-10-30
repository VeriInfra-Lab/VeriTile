import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SignedSetup

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Body
open VeriTile.Triton FP.Structural Program Direct.FPValues Setup
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def firstEntry (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) :=
  (preloop M reduction h batch row slot data metadata).val.setReg "n" .int [] (fun _ => 0)

/-- Constructive execution of the actual typed loop body with an opaque local
FP16 reduction primitive. No complete-output or successful-execution premise. -/
def iteration (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) :
    {s : State α // run (machine M reduction) (body h)
      (firstEntry M reduction h batch row slot data metadata)=some s} := by
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  cases ev <;> cases adds
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [body,active,code,Surface.source,SourceView.code,SourceView.rawCode,SourceView.active,SourceView.rawActive,SourceView.selected,Kernels.source,
      ComputeKernel.surfaceBody,Source.conditional?,Source.intRange?,
      Option.map_some,Option.getD_some,beq_self_eq_true]
    dsimp [firstEntry,preloop]
    simp (config := {maxSteps := 300000}) [Source.intRange?,Option.map,Option.getD,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,FP.Structural.store,
      State.setReg,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,entry,Data.cells,metadata]
    try rfl

#axiomsClean iteration
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Body
