import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IntegerCandidate

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IntegerCandidate
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
attribute [local irreducible] FPValue.castWord FPValue.mulWord FPValue.addWord FPValue.treeValue

private theorem range_safe_one (A : Algebra Word) (bounds : RegionName → Nat)
    (idx : RegName) (body : List ComputeStmt) (s : State Word) :
    Safety.rangeSafe A bounds idx 0 1 256 body s ↔
      Safety.traceSafeList A bounds body (s.setReg idx .nat [] (fun _ => 0)) := by
  rw [Safety.rangeSafe_unfold]
  simp only [Nat.reduceEqDiff,Nat.reduceLT,↓reduceIte]
  cases h : run A body (s.setReg idx .nat [] (fun _ => 0)) <;>
    simp [Safety.rangeSafe_unfold,h]

private theorem rangeInt_safe_one (A : Algebra Word) (bounds : RegionName → Nat)
    (idx : RegName) (body : List ComputeStmt) (s : State Word) :
    Safety.rangeIntSafe A bounds idx 0 1 256 body s ↔
      Safety.traceSafeList A bounds body (s.setReg idx .int [] (fun _ => 0)) := by
  rw [Safety.rangeIntSafe_unfold]
  simp only [Nat.reduceEqDiff,Int.reduceLT,↓reduceIte]
  cases h : run A body (s.setReg idx .int [] (fun _ => 0)) <;>
    simp [Safety.rangeIntSafe_unfold,h]

theorem safe_iff (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry) (positive : 0<g.N)
    (bounds : RegionName → Nat) (s : State Word) :
    TraceSafe (Branches.engine rest .ieeeValue plans) bounds (fullSource g.parameters) s ↔
      TraceSafe (Branches.engine rest .ieeeValue plans) bounds (unitSplitSource g.parameters) s := by
  have one := length_one g positive
  change (g.N+g.N-1)/g.N=1 at one
  by_cases sentinel : Cell.read .int (s.mem "I" (s.pids 1)) = -1
  all_goals cases he : g.even <;> cases ha : g.add
  all_goals
    simp [TraceSafe,fullSource,unitSplitSource,Geometry.parameters,he,ha,sentinel,
      ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.traceSafeList_nil,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,range_safe_one,rangeInt_safe_one,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,range,rangeInt,
      Option.bind,Option.map,Branches.engine,FPValue.backend,FPValue.formatOf,
      FPValue.typedSum,FPValue.sumValues,numeric,numericEq,numericLt,natLt,
      ofFloat,toFloat,bop,Broadcast.leftIndex,Broadcast.rightIndex,one,
      ComputeDType.eraseDType,State.setReg,Region.cast,store,-Nat.cast_mul,-Nat.cast_add,
      ←Int.natCast_mul,←Int.natCast_add]

#axiomsClean safe_iff
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IntegerCandidate
