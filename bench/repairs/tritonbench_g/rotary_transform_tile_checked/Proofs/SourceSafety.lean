import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.SourceBlockSafety
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.TypedGrid
import VeriTile.Triton.Float.StructuralSafety

/-! Access safety of the actual typed quotation. The source prefixes and
successor states are the existing execution witnesses; no replacement runner
or numerical condition is used. -/
namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceSafety
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural TypedExecution
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_regs)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The proof-only branch selection preserves safety along the actual
execution, including failed executions; it adds no source-success premise. -/
theorem source_body_safe_iff (M : Algebra α) (s : Shape)
    (bounds : RegionName → Nat) (initial : State α) :
    TraceSafe M bounds (kernel s Bool.true Bool.true) initial ↔
      Safety.traceSafeList M bounds (sourceBody s) initial := by
  have h := SourceBlocks.safe_constant_branch M bounds s.varlen
    ((kernel s Bool.true Bool.true).surfaceBody.take 4)
    (((kernel s Bool.true Bool.true).surfaceBody.drop 4).flatMap (SourceBlocks.select Bool.false))
    (((kernel s Bool.true Bool.true).surfaceBody.drop 4).flatMap (SourceBlocks.select Bool.true)) initial
  cases ht : s.offsetTensor <;>
    simpa only [TraceSafe,sourceBody,SourceBlocks.body,kernel,KernelSource.source,KernelSource.source64_64,
      KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap_cons,List.flatMap_nil,SourceBlocks.select,List.append_nil,
      ht,Bool.false_eq_true,if_false,if_true] using h

#axiomsClean source_body_safe_iff

theorem setup_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat)
    (initial : State α)
    (cu : s.varlen = Bool.true → initial.pids 1+1 < bounds "cu") :
    Safety.traceSafeList M bounds
      ((sourceBody s).take (SourceBlocks.setupCount s.varlen)) initial := by
  by_cases hv : s.varlen = Bool.true <;> by_cases ht : s.offsetTensor = Bool.true
  all_goals
    simp [kernel,KernelSource.source,KernelSource.source64_64,sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,hv,ht,
      ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,
      Safety.safeExpr,Safety.safeCompute,Safety.safeOp,Safety.safeOp_unfold,
      Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      integralFloorDiv,integralMod,ofFloat,toFloat,FloatDType.numericDType,
      FloatDType.toTileDType,ComputeDType.eraseDType,Option.bind]
  all_goals exact ⟨Nat.lt_trans (Nat.lt_succ_self _) (cu hv),cu hv⟩

theorem indices_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat)
    (initial : State α)
    (offset : s.offsetTensor = Bool.true → initial.pids 1 < bounds "offsets") :
    Safety.traceSafeList M bounds ((continuation s).take 4)
      (setup M s Bool.true Bool.true initial).val := by
  by_cases hv : s.varlen = Bool.true <;> by_cases ht : s.offsetTensor = Bool.true
  all_goals
    simp [continuation,ifBody,setup,kernel,KernelSource.source,KernelSource.source64_64,sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,hv,ht,
      ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,
      Safety.safeExpr,Safety.safeCompute,Safety.safeOp,Safety.safeOp_unfold,
      Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      integralFloorDiv,integralMod,ofFloat,toFloat,FloatDType.numericDType,
      FloatDType.toTileDType,ComputeDType.eraseDType,Option.bind]
  all_goals exact offset ht

/-- Window bounds for the full rotary prefix and the two coefficient tables.
Only the source mask decides whether a lane needs backing. -/
structure AccessBounds (s : Shape) (u : State α) (bounds : RegionName → Nat) : Prop where
  data : ∀ lane col, ProgramModel.active s u lane col Bool.true →
    ProgramModel.dataOffset s u lane col < bounds "x" ∧
      ProgramModel.dataOffset s u lane col < bounds (outputRegion s)
  coefficient : ∀ lane pair, ProgramModel.hasCoefficient s u lane pair →
    ProgramModel.coefficientOffset s u lane pair < bounds "cos" ∧
      ProgramModel.coefficientOffset s u lane pair < bounds "sin"

theorem loads_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat)
    (initial : State α) (bound : AccessBounds s initial bounds) :
    Safety.traceSafeList M bounds ((payload s).take (loadsCount s))
      (indices M s initial).val := by
  have hcos (lane pair : Nat) (hr : (ProgramModel.row s initial lane : Int)+ProgramModel.shift s initial<s.coefficientRows)
      (hp : pair<s.half) := (bound.coefficient lane pair ⟨hr,hp⟩).1
  have hsin (lane pair : Nat) (hr : (ProgramModel.row s initial lane : Int)+ProgramModel.shift s initial<s.coefficientRows)
      (hp : pair<s.half) := (bound.coefficient lane pair ⟨hr,hp⟩).2
  have hx (lane col : Nat) (hr : (ProgramModel.row s initial lane : Int)<ProgramModel.sequenceLength s initial)
      (hc : col<2*s.half) := (bound.data lane col ⟨hr,hc⟩).1
  have hhigh (lane pair : Nat) (hr : (ProgramModel.row s initial lane : Int)<ProgramModel.sequenceLength s initial)
      (hp : pair<s.half) : ProgramModel.dataOffset s initial lane pair+s.half<bounds "x" := by
    simpa only [ProgramModel.dataOffset,Nat.add_assoc] using hx lane (pair+s.half) hr (by omega)
  by_cases hv : s.varlen = Bool.true <;> by_cases ht : s.offsetTensor = Bool.true <;>
    by_cases hi : s.interleaved = Bool.true
  all_goals
    simp (config := { maxSteps := 80000 }) [payload,loadsCount,branchBody,continuation,ifBody,
      indices,setup,kernel,KernelSource.source,KernelSource.source64_64,sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,hv,ht,hi,
      ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,
      Safety.safeExpr,Safety.safeCompute,Safety.safeOp,Safety.safeOp_unfold,
      Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      integralFloorDiv,integralMod,ofFloat,toFloat,FloatDType.numericDType,
      FloatDType.toTileDType,ComputeDType.eraseDType,Option.bind]
  all_goals
    simp [ProgramModel.row,ProgramModel.shift,ProgramModel.sequenceLength,
      ProgramModel.dataOffset,ProgramModel.dataBase,ProgramModel.coefficientOffset,hv,ht,
      Int.natCast_add,Int.natCast_mul] at hcos hsin hx hhigh
  all_goals refine ⟨?_,?_,?_,?_⟩
  all_goals intro lane col hr hc
  all_goals first
    | exact hcos lane.val (col.val/2) hr hc
    | exact hcos lane.val col.val hr hc
    | exact hsin lane.val (col.val/2) hr hc
    | exact hsin lane.val col.val hr hc
    | exact hx lane.val col.val hr hc
    | exact hx lane.val col.val hr (by omega)
    | exact hx lane.val (col.val+(col.val+1)%2*2-1) hr hc
    | exact Nat.lt_of_le_of_lt (Nat.le_add_right _ _) (hhigh lane.val col.val hr hc)
    | exact hhigh lane.val col.val hr hc

end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceSafety
