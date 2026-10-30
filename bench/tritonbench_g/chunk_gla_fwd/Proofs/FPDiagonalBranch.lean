import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPDiagonalLoop
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem not_decide_le (a b : Nat) : (!decide (a ≤ b))=decide (b<a) := by
  by_cases h : b<a
  · simp [h,Nat.not_le.mpr h]
  · simp [h,Nat.le_of_not_gt h]

private theorem if_lt_min (a b : Nat) : (if a<b then a else b)=min a b := by
  by_cases h : a<b
  · simp [h,Nat.min_eq_left (Nat.le_of_lt h)]
  · simp [h,Nat.min_eq_right (Nat.le_of_not_gt h)]

def prologue (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (s : State α) :
    {t : State α // run (engine M) (branch split c scale).before s=some t} := by
  cases split with
  | «false» =>
    refine ⟨?_,?_⟩
    swap
    simp [branch,kernel,SourceBridge.intraNew,TypedIntra.original,TypedIntra.source,TypedIntra.fp32Original,IntraGrid.layout,IntraGrid.regions,
      ComputeKernel.surfaceBody,Source.conditional?,Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
    rfl
  | «true» =>
    refine ⟨?_,?_⟩
    swap
    simp [branch,kernel,SourceBridge.splitNew,TypedSplit.original,TypedSplit.source,TypedSplit.fp32Original,SplitGrid.layout,SplitGrid.regions,
      ComputeKernel.surfaceBody,Source.conditional?,Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
    rfl

theorem condition_eval (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (s : State α) :
    FP.Structural.evalOp (engine M) none (branch split c scale).condition (prologue M split c scale s).val=
      some (fun _ => decide (enabled split c s)) := by
  cases split
  all_goals simp [branch,kernel,SourceBridge.intraNew,SourceBridge.splitNew,TypedIntra.original,TypedIntra.source,TypedIntra.fp32Original,
    TypedSplit.original,TypedSplit.source,TypedSplit.fp32Original,IntraGrid.layout,IntraGrid.regions,SplitGrid.layout,SplitGrid.regions,
    ComputeKernel.surfaceBody,Source.conditional?,evalOp_unfold,numeric,bop,prologue,natLe,natLt,enabled,rowStart,chunk,block]
  all_goals
    funext i
    exact not_decide_le _ _

theorem stop_eval (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (s : State α) :
    FP.Structural.evalOp (engine M) none (parts split c scale).stop (entryExecution M split c scale s).val=
      some (fun _ => columns split c s) := by
  cases split
  all_goals simp [parts,branch,kernel,SourceBridge.intraNew,SourceBridge.splitNew,TypedIntra.original,TypedIntra.source,TypedIntra.fp32Original,
    TypedSplit.original,TypedSplit.source,TypedSplit.fp32Original,IntraGrid.layout,IntraGrid.regions,SplitGrid.layout,SplitGrid.regions,
    ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    evalOp_unfold,numeric,bop,entryExecution,natLe,natLt,numericLt,columns,chunk,block]
  all_goals
    funext i
    exact if_lt_min _ _

theorem source_parts (split : Bool) (c : Layout) (scale : ℝ) : (kernel split c scale).surfaceBody = (branch split c scale).before ++
    [.ifThen (branch split c scale).condition ((parts split c scale).before ++
      [.forRangeDyn "j" (parts split c scale).start (parts split c scale).stop (parts split c scale).stride (parts split c scale).body])] := by
  cases split <;> rfl

#axiomsClean condition_eval
#axiomsClean stop_eval
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
