import VeriTile.Triton.Float.Loop
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Source

/-! Proof-only control-flow decomposition of the copied source. The tile body
stays an opaque statement-list parameter while control operations are compared;
no floating computation or memory operation is unfolded by the bridge. -/
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.Control
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.PowScalarTensor WholeTile
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open SignedView StructuralGrid StructuralControl
noncomputable section
set_option maxRecDepth 6000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

private def computeAssignment : Stmt → ComputeStmt
  | .assign d sh name value => .assign d sh name (.alg value)
  | st => .alg st

def sourcePrefixCode (n B : Nat) : List ComputeStmt :=
  [computeAssignment (.assign .nat [] "pid" (.programId 0)),
   computeAssignment (SourceForRange.numTilesStmt n B)]

def oneBody (body : List ComputeStmt) (n B : Nat) : List ComputeStmt :=
  sourcePrefixCode n B ++ computeAssignment (.assign .nat [] "tile_id" (.ref .nat [] "pid"))::body

def gridBody (body : List ComputeStmt) (n B tiles : Nat) : List ComputeStmt :=
  sourcePrefixCode n B ++ [computeAssignment (.assign .nat [] "num_ctas" (.numPrograms 0)),
    .forRange "j" 0 tiles 1 (computeAssignment SourceForRange.tileIdStmt::body)]

def selected (c : Parameters) (candidate : Bool) : List ComputeStmt :=
  (sourceKernel c candidate).surfaceBody.take 2 ++
    (if c.oneTile then PastedTile.oneTile c candidate else PastedTile.grid c candidate)

theorem selected_layout (c : Parameters) (candidate : Bool) :
    selected c candidate = if c.oneTile then oneBody (perTile c candidate) c.n c.B
      else gridBody (perTile c candidate) c.n c.B c.tiles := by
  cases candidate <;> cases branch : c.oneTile
  all_goals simp [selected,oneBody,gridBody,sourcePrefixCode,computeAssignment,perTile,
    PastedTile.body,PastedTile.oneTile,PastedTile.grid,PastedTile.iterations,
    sourceKernel,kernel,commutedKernel,branch,oneTileKernel,oneTileCommuted,
    gridStrideKernel,gridStrideCommuted,originalKernel,commutedBodyPastedKernel,
    SourceForRange.tileIdStmt,SourceForRange.numTilesStmt,ComputeKernel.surfaceBody]

private theorem constant_branch {α : Type} [Inhabited α] (M : Algebra α)
    (flag : Bool) (pre yes no : List ComputeStmt) (s : State α) :
    run M (pre++[.ifThenElse (.constBool flag) yes no]) s =
      run M (pre++(if flag then yes else no)) s := by
  rw [run_append,run_append]
  apply congrArg
  funext t
  cases flag
  · simp only [Bool.false_eq_true,if_false,run,step,evalOp_unfold,bind,pure,Option.bind]
    cases run M no t <;> rfl
  · simp only [Bool.true_eq,if_true,run,step,evalOp_unfold,bind,pure,Option.bind]
    cases run M yes t <;> rfl

private theorem source_layout (c : Parameters) (candidate : Bool) :
    (sourceKernel c candidate).surfaceBody = (sourceKernel c candidate).surfaceBody.take 2 ++
      [.ifThenElse (.constBool c.oneTile) (PastedTile.oneTile c candidate) (PastedTile.grid c candidate)] := by
  cases candidate <;> cases branch : c.oneTile <;>
    simp [sourceKernel,kernel,commutedKernel,branch,oneTileKernel,gridStrideKernel,
      oneTileCommuted,gridStrideCommuted,originalKernel,commutedBodyPastedKernel,ComputeKernel.surfaceBody,
      PastedTile.oneTile,PastedTile.grid]

theorem source_run {α : Type} [Inhabited α] (M : Algebra α) (c : Parameters)
    (candidate : Bool) (s : State α) :
    FP.Structural.exec M (sourceKernel c candidate) s = run M (selected c candidate) s := by
  have exec_surface (k : ComputeKernel) : FP.Structural.exec M k s=run M k.surfaceBody s := by
    cases k; rfl
  rw [exec_surface,source_layout,constant_branch]
  rfl

private theorem range_assign {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (start stop stride : Nat) (name : RegName)
    (e : Op dtype shape) (body : List ComputeStmt) (s : State α) :
    FP.Structural.range M idx start stop stride (.assign dtype shape name (.alg e)::body) s =
      FP.Structural.range M idx start stop stride (.alg (.assign dtype shape name e)::body) s := by
  apply range_congr_on_reachable
  intro cur t _ _ _
  simp only [run,step,stepAlg,evalExpr]

theorem one_run {α : Type} [Inhabited α] (M : Algebra α) (body : List ComputeStmt)
    (n B : Nat) (s : State α) :
    run M (oneBody body n B) s = run M (StructuralControl.oneTileBody body n B) s := by
  simp only [oneBody,sourcePrefixCode,computeAssignment,StructuralControl.oneTileBody,
    SourceForRange.numTilesStmt,List.cons_append,List.nil_append,run,step,stepAlg,evalExpr]

theorem grid_run {α : Type} [Inhabited α] (M : Algebra α) (body : List ComputeStmt)
    (n B tiles : Nat) (s : State α) :
    run M (gridBody body n B tiles) s = run M (StructuralControl.gridBody body n B tiles) s := by
  simp only [gridBody,sourcePrefixCode,computeAssignment,StructuralControl.gridBody,
    StructuralControl.loopBody,SourceForRange.tileIdStmt,SourceForRange.numTilesStmt,
    List.cons_append,List.nil_append,run,step,stepAlg,evalExpr,evalOp_unfold,
    bind,pure,Option.bind,numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rw [range_assign]

#axiomsClean selected_layout
#axiomsClean source_run
#axiomsClean one_run
#axiomsClean grid_run
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.Control
