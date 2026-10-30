import bench.tritonbench_g.rmsnorm_triton.Proofs.KernelPublic
import bench.tritonbench_g.rmsnorm_triton.Proofs.Mathematical_Candidate

namespace VeriTile.Bench.Repairs.RmsnormTriton.Source
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton
open Full.ScratchRms Full.Prefix Full.PostStep Full.Wb
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- Exponent two becomes square only in the independent Real interpretation. -/
theorem variance_assignment (B : Nat) (s : BlockState) :
    stepStmts
      [.assign .real [B] "var" (.add .real (.consSame .nil)
        (.ref .real [B] "var")
        (.libdevicePowi .scalarR (.ref .real [B] "x") (.constInt 2)))] s =
    stepStmts
      [.assign .real [B] "var" (.add .real (.consSame .nil)
        (.ref .real [B] "var")
        (.mul .real (.consSame .nil)
          (.castFloat .real .real (.ref .real [B] "x"))
          (.castFloat .real .real (.ref .real [B] "x"))))] s := by
  simp only [stepStmts,stepStmt,evalOp_add,powi_square_eval,evalOp_mul]
  simp only [real_cast_eval]

theorem variance_run (X : RegionName) (sxk N B : Nat) (s : BlockState) :
    stepStmts (powiVarBody X sxk N B) s=stepStmts (varBody X sxk N B) s := by
  have body : varBody X sxk N B=(varBody X sxk N B).take 3 ++
      [.assign .real [B] "var" (.add .real (.consSame .nil)
        (.ref .real [B] "var")
        (.mul .real (.consSame .nil)
          (.castFloat .real .real (.ref .real [B] "x"))
          (.castFloat .real .real (.ref .real [B] "x"))))] := rfl
  conv_rhs => rw [body]
  simp only [powiVarBody,ProgramRewrite.append,variance_assignment]

theorem prefix_run (v : Bool) (sxb sxm : Nat) (s : BlockState) :
    stepStmts (indexPrefix v sxb sxm) s=stepStmts (indexPrefix Bool.false sxb sxm) s := by
  cases v
  · rfl
  · simp [indexPrefix,stepStmts,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.mul,Nat.mul_comm]

theorem math_run (v : Bool) (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat)
    (eps : ℝ) (s : BlockState) :
    exec (mathKernel v X W O sxb sxm sxk sw sob som sok N B eps) s=
      exec (rmsnorm_triton X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel s := by
  have loop : ∀ t : BlockState,
      stepStmts [.forRange "block_n_start_idx" 0 N B (powiVarBody X sxk N B)] t=
      stepStmts [.forRange "block_n_start_idx" 0 N B (varBody X sxk N B)] t := by
    intro t
    simp only [stepStmts,stepStmt]
    rw [ProgramRewrite.range_congr _ _ _ _ _ _ (variance_run X sxk N B)]
  have prefixBody : prefixStmts X sxb sxm B=indexPrefix Bool.false sxb sxm ++
      (prefixStmts X sxb sxm B).drop 3 := rfl
  unfold exec
  rw [math_body,Full.Final.body_decomp]
  conv_rhs => rw [prefixBody]
  simp only [ProgramRewrite.append,loop,prefix_run]

theorem original_run (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat)
    (eps : ℝ) (s : BlockState) :
    exec (kernel X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel s=
      exec (rmsnorm_triton X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel s := by
  rw [kernel_source,typed_projection]
  exact math_run _ X W O sxb sxm sxk sw sob som sok N B eps s

theorem candidate_run (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat)
    (eps : ℝ) (s : BlockState) :
    exec (indexCommutedKernel X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel s=
      exec (kernel X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel s := by
  rw [candidate_source,kernel_source,typed_projection,typed_projection,math_run,math_run]

#axiomsClean original_run
#axiomsClean candidate_run
end VeriTile.Bench.Repairs.RmsnormTriton.Source
