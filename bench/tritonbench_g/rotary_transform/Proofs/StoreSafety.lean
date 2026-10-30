import bench.tritonbench_g.rotary_transform.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceSafety
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural TypedExecution
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_regs)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem stores_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat)
    (initial : State α) (bound : AccessBounds s initial bounds) :
    Safety.traceSafeList M bounds ((payload s).drop (loadsCount s))
      (loads M s initial).val := by
  have ho (lane col : Nat) (hr : (ProgramModel.row s initial lane : Int)<ProgramModel.sequenceLength s initial)
      (hc : col<2*s.half) := (bound.data lane col ⟨hr,hc⟩).2
  have hhigh (lane pair : Nat) (hr : (ProgramModel.row s initial lane : Int)<ProgramModel.sequenceLength s initial)
      (hp : pair<s.half) : ProgramModel.dataOffset s initial lane pair+s.half<bounds (outputRegion s) := by
    simpa only [ProgramModel.dataOffset,Nat.add_assoc] using ho lane (pair+s.half) hr (by omega)
  by_cases hv : s.varlen = Bool.true <;> by_cases ht : s.offsetTensor = Bool.true <;>
    by_cases hi : s.interleaved = Bool.true <;> by_cases hc : s.conjugate = Bool.true
  all_goals
    simp (config := { maxSteps := 150000 }) [payload,loadsCount,branchBody,continuation,ifBody,
      loads,indices,setup,kernel,KernelSource.source,KernelSource.source64_64,hv,ht,hi,hc,
      ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,
      Safety.safeExpr,Safety.safeCompute,Safety.safeOp,Safety.safeOp_unfold,
      Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,store,↓fold_writes_regs,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      integralFloorDiv,integralMod,ofFloat,toFloat,FloatDType.numericDType,
      FloatDType.toTileDType,ComputeDType.eraseDType,Option.bind]
  all_goals
    simp [ProgramModel.row,ProgramModel.sequenceLength,ProgramModel.dataOffset,
      ProgramModel.dataBase,hv,Int.natCast_add,Int.natCast_mul] at ho hhigh
  all_goals repeat' apply And.intro
  all_goals intro lane col hr hd
  all_goals first
    | exact ho lane.val col.val hr hd
    | exact ho lane.val col.val hr (by omega)
    | exact Nat.lt_of_le_of_lt (Nat.le_add_right _ _) (hhigh lane.val col.val hr hd)
    | exact hhigh lane.val col.val hr hd

#axiomsClean stores_safe
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceSafety
