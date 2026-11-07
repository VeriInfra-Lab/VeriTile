import bench.tritonbench_g.pow_scalar_tensor.Kernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeTile
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.PastedTile
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def oneTile (c : Parameters) (candidate : Bool) : List ComputeStmt :=
  match (if candidate then oneTileCommuted c else oneTileKernel c).surfaceBody[2]? with
  | some (ComputeStmt.ifThenElse _ yes _) => yes
  | _ => []
def grid (c : Parameters) (candidate : Bool) : List ComputeStmt :=
  match (if candidate then gridStrideCommuted c else gridStrideKernel c).surfaceBody[2]? with
  | some (ComputeStmt.ifThenElse _ _ no) => no
  | _ => []
def iterations (c : Parameters) (candidate : Bool) : List ComputeStmt :=
  match (grid c candidate)[1]? with
  | some (ComputeStmt.forRange _ _ _ _ body) | some (ComputeStmt.forRangeDyn _ _ _ _ body)
  | some (ComputeStmt.forRangeInt _ _ _ _ body) | some (ComputeStmt.forLoop _ _ body) => body
  | _ => []
def body (c : Parameters) (candidate : Bool) : List ComputeStmt :=
  (if c.oneTile then oneTile c candidate else iterations c candidate).drop 1

def execution {α : Type} [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool)
    (T : Nat) (s : State α) (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) :
    {t : State α // run M (body c candidate) s=some t} := by
  cases candidate <;> cases branch : c.oneTile
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [body,branch,oneTile,grid,iterations,oneTileKernel,oneTileCommuted,gridStrideKernel,
      gridStrideCommuted,originalKernel,commutedBodyPastedKernel,ComputeKernel.surfaceBody,
      run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      numeric,numericLt,numericLe,numericEq,natLt,FP.Structural.bop,store,ht,
      ComputeDType.eraseDType,FloatDType.comparableDType,Region.cast,ofFloat,toFloat,
      Broadcast.leftIndex,Broadcast.rightIndex,Int.not_ofNat_neg]
    rfl

@[simp] theorem masked_regs {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).regs=s.regs :=
  (VeriTile.Bench.TritonBenchG.PowScalarTensor.WholeTile.scatter_control r off value active lanes s).1
@[simp] theorem masked_undef {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).undef=s.undef :=
  (VeriTile.Bench.TritonBenchG.PowScalarTensor.WholeTile.scatter_control r off value active lanes s).2.1

#axiomsClean execution
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.PastedTile
