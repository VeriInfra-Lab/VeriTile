import bench.tritonbench_g.relu_strided_buffer.Kernels
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralTile
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.PastedTile
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def oneTile (c : Parameters) : List ComputeStmt :=
  match (oneTileKernel c).surfaceBody[2]? with
  | some (ComputeStmt.ifThenElse _ yes _) => yes
  | _ => []
def grid (c : Parameters) : List ComputeStmt :=
  match (gridStrideKernel c).surfaceBody[2]? with
  | some (ComputeStmt.ifThenElse _ _ no) => no
  | _ => []
def iterations (c : Parameters) : List ComputeStmt :=
  match (grid c)[1]? with
  | some (ComputeStmt.forRange _ _ _ _ body) | some (ComputeStmt.forRangeDyn _ _ _ _ body)
  | some (ComputeStmt.forRangeInt _ _ _ _ body) | some (ComputeStmt.forLoop _ _ body) => body
  | _ => []
def body (c : Parameters) : List ComputeStmt :=
  (if c.oneTile then oneTile c else iterations c).drop 1

theorem body_length (c : Parameters) : (body c).length=14 := by
  cases h : c.oneTile <;> simp [body,h,oneTile,grid,iterations,oneTileKernel,gridStrideKernel,
    originalKernel,ComputeKernel.surfaceBody]

def execution {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    { t : State α // run M (body c) s=some t } := by
  cases branch : c.oneTile
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [body,branch,oneTile,grid,iterations,oneTileKernel,gridStrideKernel,originalKernel,
      ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,numericLt,numericLe,numericEq,natLt,FP.Structural.bop,store,hc,ht,
      ComputeDType.eraseDType,FloatDType.comparableDType,Region.cast,ofFloat,toFloat,
      Option.bind,Broadcast.leftIndex,Broadcast.rightIndex,Int.not_ofNat_neg]
    rfl

@[simp] theorem masked_regs {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).regs=s.regs :=
  (VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralTile.scatter_control r off value active lanes s).1

@[simp] theorem masked_undef {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).undef=s.undef :=
  (VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralTile.scatter_control r off value active lanes s).2.1

#axiomsClean body_length
#axiomsClean execution
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.PastedTile
