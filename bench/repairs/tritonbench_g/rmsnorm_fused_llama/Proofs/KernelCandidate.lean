import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.FPGrid
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RealGrid
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Candidate
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Precision

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelCandidate
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- The complete candidate preserves every FP operation, backend invocation,
unspecified inactive payload, full state and failure behavior. -/
theorem typed {α : Type} [Inhabited α] (A : Algebra α) (X Y W : RegionName)
    (N B : Nat) (eps : ℝ) (s : State α) :
    FP.Structural.exec A (indexCommutedKernel X Y W N B eps) s=
      FP.Structural.exec A (kernel X Y W N B eps) s := by
  let original := (kernel X Y W N B eps).surfaceBody
  let candidate := (indexCommutedKernel X Y W N B eps).surfaceBody
  have pref : run A (candidate.take 2) s=run A (original.take 2) s := by
    simp [candidate,original,indexCommutedKernel,kernel,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,State.setReg,Option.bind]
    delta bop
    simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  change run A (candidate.take 2 ++ original.drop 2) s=run A (original.take 2 ++ original.drop 2) s
  rw [run_append,run_append,pref]

def grid {α : Type} [Inhabited α] (A : Algebra α) (X Y W : RegionName)
    (rows N B : Nat) (eps : ℝ) (initial : State α) : Nat → Option (State α)
  | 0=>some initial
  | c+1=>(grid A X Y W rows N B eps initial c).bind
      (fun s=>FP.Structural.exec A (indexCommutedKernel X Y W N B eps) (Grid.setup rows c s))

theorem grid_eq {α : Type} [Inhabited α] (A : Algebra α) (X Y W : RegionName)
    (rows N B : Nat) (eps : ℝ) (initial : State α) (count : Nat) :
    grid A X Y W rows N B eps initial count=Grid.run A X Y W rows N B eps initial count := by
  induction count with
  | zero=>rfl
  | succ c ih=>simp only [grid,Grid.run,ih,typed]

open VeriTile.Bench.TritonBenchG.RmsnormFusedLlama in
theorem real (X Y W : RegionName) (N B : Nat) (eps : ℝ) (s : BlockState) :
    Triton.exec (indexCommutedKernel X Y W N B eps).eraseDType.toAlgKernel s=
      Triton.exec (kernel X Y W N B eps).eraseDType.toAlgKernel s := by
  rw [candidate_eq,source_eq]
  change Triton.exec (typedKernel .fp32 .fp32 .fp32 Bool.true X Y W N N B eps).eraseDType.toAlgKernel s=
    Triton.exec (typedKernel .fp32 .fp32 .fp32 Bool.false X Y W N N B eps).eraseDType.toAlgKernel s
  rw [typed_projection,typed_projection]
  have pref : stepStmts (pointerPrefix Bool.true Y N) s=stepStmts (pointerPrefix Bool.false Y N) s := by
    simp [pointerPrefix,stepStmts,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.mul,Nat.mul_comm]
  unfold Triton.exec
  rw [←List.take_append_drop 2 (mathKernel Bool.true X Y W N N B eps).body,
    ←List.take_append_drop 2 (mathKernel Bool.false X Y W N N B eps).body]
  have tail : (mathKernel Bool.true X Y W N N B eps).body.drop 2=
    (mathKernel Bool.false X Y W N N B eps).body.drop 2 := rfl
  rw [prefix_source,prefix_source,ProgramRewrite.append,ProgramRewrite.append,pref,tail]

def realGrid (X Y W : RegionName) (rows N B : Nat) (eps : ℝ) (initial : BlockState) : Nat → Option BlockState
  | 0=>some initial
  | c+1=>(realGrid X Y W rows N B eps initial c).bind
      (fun s=>Triton.exec (indexCommutedKernel X Y W N B eps).eraseDType.toAlgKernel (RealGrid.setup rows c s))

theorem real_grid_eq (X Y W : RegionName) (rows N B : Nat) (eps : ℝ) (initial : BlockState) (count : Nat) :
    realGrid X Y W rows N B eps initial count=RealGrid.run X Y W rows N B eps initial count := by
  induction count with
  | zero=>rfl
  | succ c ih=>simp only [realGrid,RealGrid.run,ih,real]

#axiomsClean typed
#axiomsClean grid_eq
#axiomsClean real
#axiomsClean real_grid_eq
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelCandidate
