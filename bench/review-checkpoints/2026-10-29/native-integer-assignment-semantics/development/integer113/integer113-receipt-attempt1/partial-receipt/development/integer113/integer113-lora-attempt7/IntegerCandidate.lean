import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.SourceConnection
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.PairIO

/-! Ordinary `split_n_length = 1` is an integer tensor. The candidate's local
register types may differ from the cdiv source. Compare actual execution after
forgetting private registers; retain memory, program IDs and undef behavior. -/
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IntegerCandidate
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
attribute [local irreducible] FPValue.castWord FPValue.mulWord FPValue.addWord FPValue.treeValue

def forgetLocals (s : State Word) : State Word :=
  {s with regs := fun _ _ _ => none}

@[simp] theorem forget_setReg (s : State Word) (n : RegName) (d : TileDType)
    (sh : TileShape) (v : Values Word d sh) :
    forgetLocals (s.setReg n d sh v)=forgetLocals s := rfl

private theorem range_one (A : Algebra Word) (idx : RegName)
    (body : List ComputeStmt) (s : State Word) :
    range A idx 0 1 256 body s=run A body (s.setReg idx .nat [] (fun _ => 0)) := by
  rw [range]
  simp only [Nat.reduceEqDiff, Nat.reduceLT, ↓reduceIte]
  cases h : run A body (s.setReg idx .nat [] (fun _ => 0)) with
  | none => rfl
  | some t => simp [range,h]

private theorem rangeInt_one (A : Algebra Word) (idx : RegName)
    (body : List ComputeStmt) (s : State Word) :
    rangeInt A idx 0 1 256 body s=run A body (s.setReg idx .int [] (fun _ => 0)) := by
  rw [rangeInt]
  simp only [Nat.reduceEqDiff, Int.reduceLT, ↓reduceIte]
  cases h : run A body (s.setReg idx .int [] (fun _ => 0)) with
  | none => rfl
  | some t => simp [rangeInt,h]

theorem observed_run (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry) (positive : 0<g.N)
    (s : State Word) :
    (exec (Branches.engine rest .ieeeValue plans) (fullSource g.parameters) s).map forgetLocals =
      (exec (Branches.engine rest .ieeeValue plans) (unitSplitSource g.parameters) s).map forgetLocals := by
  have one := length_one g positive
  change (g.N+g.N-1)/g.N=1 at one
  by_cases sentinel : Cell.read .int (s.mem "I" (s.pids 1)) = -1
  all_goals cases he : g.even <;> cases ha : g.add
  all_goals
    simp [FP.Structural.exec,fullSource,unitSplitSource,Geometry.parameters,he,ha,sentinel,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,Option.map,Branches.engine,FPValue.backend,FPValue.formatOf,
      FPValue.typedSum,FPValue.sumValues,numeric,numericEq,numericLt,natLt,ofFloat,toFloat,bop,Broadcast.leftIndex,
      Broadcast.rightIndex,one,range_one,rangeInt_one,forgetLocals,
      State.setReg,Region.cast,store, -Nat.cast_mul, -Nat.cast_add,
      ←Int.natCast_mul,←Int.natCast_add]

#axiomsClean observed_run
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IntegerCandidate
