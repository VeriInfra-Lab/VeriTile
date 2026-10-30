import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeGrid
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.RuntimeCandidate
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
/-- The complete candidate preserves every FP operation, backend invocation,
unspecified inactive payload, full state and failure behavior. -/
theorem typed {α : Type} [Inhabited α] (A : Algebra α) (X Y W : RegionName)
    (N B : Nat) (s : State α) :
    FP.Structural.exec A (runtimeCandidate X Y W N B) s=
      FP.Structural.exec A (runtimeKernel X Y W N B) s := by
  let original := (runtimeKernel X Y W N B).surfaceBody
  let candidate := (runtimeCandidate X Y W N B).surfaceBody
  have pref : run A (candidate.take 2) s=run A (original.take 2) s := by
    simp [candidate,original,runtimeCandidate,runtimeKernel,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,State.setReg,Option.bind]
    delta bop
    simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  change run A (candidate.take 2 ++ original.drop 2) s=run A (original.take 2 ++ original.drop 2) s
  rw [run_append,run_append,pref]

def grid {α : Type} [Inhabited α] (A : Algebra α) (X Y W : RegionName)
    (rows N B : Nat) (eps : α) (initial : State α) : Nat → Option (State α)
  | 0=>some initial
  | c+1=>(grid A X Y W rows N B eps initial c).bind
      (fun s=>FP.Structural.exec A (runtimeCandidate X Y W N B) ((RuntimeGrid.setup rows c s).setReg "eps" .real [] (fun _=>eps)))

theorem grid_eq {α : Type} [Inhabited α] (A : Algebra α) (X Y W : RegionName)
    (rows N B : Nat) (eps : α) (initial : State α) (count : Nat) :
    grid A X Y W rows N B eps initial count=RuntimeGrid.run A X Y W rows N B eps initial count := by
  induction count with
  | zero=>rfl
  | succ c ih=>simp only [grid,RuntimeGrid.run,ih,typed]

#axiomsClean typed
#axiomsClean grid_eq
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.RuntimeCandidate
