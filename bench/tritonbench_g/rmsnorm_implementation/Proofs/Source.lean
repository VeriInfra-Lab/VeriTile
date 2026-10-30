import bench.tritonbench_g.rmsnorm_implementation.Kernels
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Mathematical_Candidate

namespace VeriTile.Bench.Repairs.RmsnormImplementation.Source
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem original (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    kernel X W O sxb sxm sxk sw sob som sok N B eps =
      typedKernel .fp32 .fp32 .fp32 Bool.false X W O sxb sxm sxk sw sob som sok N B eps := rfl

theorem commuted (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    indexCommutedKernel X W O sxb sxm sxk sw sob som sok N B eps =
      typedKernel .fp32 .fp32 .fp32 Bool.true X W O sxb sxm sxk sw sob som sok N B eps := rfl

/-- Remove only a Real-to-Real cast in the mathematical projection. -/
theorem uncast_var_run (X : RegionName) (sxk N B : Nat) (s : BlockState) :
    stepStmts (uncastVarBody X sxk N B) s = stepStmts (rmsVarLoopBody X sxk N B) s := by
  have cast : ∀ t : BlockState,
      stepStmts [.assign .real [B] "xf" (.ref .real [B] "x")] t =
      stepStmts [.assign .real [B] "xf" (.castFloat .real .real (.ref .real [B] "x"))] t := by
    intro t
    simp [stepStmts, stepStmt, evalOp.eq_def, FloatDType.cast]
    cases t.regs .real [B] "x" <;> rfl
  have body : rmsVarLoopBody X sxk N B = (rmsVarLoopBody X sxk N B).take 3 ++
      [.assign .real [B] "xf" (.castFloat .real .real (.ref .real [B] "x"))] ++
      (rmsVarLoopBody X sxk N B).drop 4 := rfl
  conv_rhs => rw [body]
  simp only [uncastVarBody, ProgramRewrite.append, cast]

theorem uncast_run (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat)
    (eps : ℝ) (s : BlockState) :
    exec (uncastMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel s =
      exec (rmsnorm_implementation X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel s := by
  have body : (uncastMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel.body =
      rmsVarPreLoop sxb sxm B ++
      [.forRange "block_n_strart_ptr" 0 N B (uncastVarBody X sxk N B)] ++
      rmsStdPostLoop N B eps ++
      [.forRange "block_n_strart_ptr" 0 N B (rmsOutLoopBody X W O sxk sw sob som sok N B)] := by
    simp [uncastMathKernel, rmsVarPreLoop, uncastVarBody, rmsVarLoopBody,
      rmsStdPostLoop, rmsOutLoopBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    repeat' constructor
  have loop : ∀ t : BlockState,
      stepStmts [.forRange "block_n_strart_ptr" 0 N B (uncastVarBody X sxk N B)] t =
      stepStmts [.forRange "block_n_strart_ptr" 0 N B (rmsVarLoopBody X sxk N B)] t := by
    intro t
    simp only [stepStmts, stepStmt]
    rw [ProgramRewrite.range_congr _ _ _ _ _ _ (uncast_var_run X sxk N B)]
  unfold exec
  rw [body, rmsnorm_implementation_toAlg_body]
  simp only [ProgramRewrite.append, loop]

theorem original_run (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat)
    (eps : ℝ) (s : BlockState) :
    exec (kernel X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel s =
      exec (rmsnorm_implementation X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel s := by
  rw [original, typed_projection]
  exact uncast_run X W O sxb sxm sxk sw sob som sok N B eps s

theorem commuted_run (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat)
    (eps : ℝ) (s : BlockState) :
    exec (indexCommutedKernel X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel s =
      exec (kernel X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel s := by
  rw [commuted, original, typed_projection, typed_projection]
  unfold exec
  rw [← List.take_append_drop 3 (mathKernel Bool.true X W O sxb sxm sxk sw sob som sok N B eps).body,
    ← List.take_append_drop 3 (mathKernel Bool.false X W O sxb sxm sxk sw sob som sok N B eps).body]
  have tail : (mathKernel Bool.true X W O sxb sxm sxk sw sob som sok N B eps).body.drop 3 =
      (mathKernel Bool.false X W O sxb sxm sxk sw sob som sok N B eps).body.drop 3 := rfl
  simp [ProgramRewrite.append, prefix_source, tail, indexPrefix,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

#axiomsClean original
#axiomsClean commuted
#axiomsClean original_run
#axiomsClean commuted_run
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RmsnormImplementation.Source
