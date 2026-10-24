import bench.tritonbench_g.rmsnorm_triton.Proofs.Source
import bench.tritonbench_g.rmsnorm_triton.Proofs.Real
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton
open VeriTile.Triton
open Full.ScratchRms Full.Prefix Full.PostStep Full.Wb
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- The source variance body retains integer power after precision erasure. -/
def powiVarBody (X : RegionName) (sxk N B : Nat) : List Stmt :=
  (varBody X sxk N B).take 3 ++
    [.assign .real [B] "var" (.add .real (.consSame .nil)
      (.ref .real [B] "var")
      (.libdevicePowi .scalarR (.ref .real [B] "x") (.constInt 2)))]

def indexPrefix (v : Bool) (sxb sxm : Nat) : List Stmt :=
  [.assign .nat [] "pid_batch" (.programId 0),
   .assign .nat [] "pid_m" (.programId 1),
   .assign .nat [] "offs_m"
     (.add .nat .nil
       (if v then .mul .nat .nil (.constNat sxb) (.ref .nat [] "pid_batch")
        else .mul .nat .nil (.ref .nat [] "pid_batch") (.constNat sxb))
       (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat sxm)))]

theorem math_body (v : Bool) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (mathKernel v X W O sxb sxm sxk sw sob som sok N B eps).body =
      indexPrefix v sxb sxm ++ (prefixStmts X sxb sxm B).drop 3 ++
      [.forRange "block_n_start_idx" 0 N B (powiVarBody X sxk N B)] ++
      postStmts N B eps ++
      [.forRange "block_n_start_idx" 0 N B (wbBody X W O sxb sxm sxk sw sob som sok N B)] := by
  cases v <;> simp [mathKernel, typedKernel, originalTyped, indexPrefix, powiVarBody,
    prefixStmts, varBody, postStmts, wbBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
    ComputeKernel.eraseDType, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]

/-- Exact-real cast identity, used only for the historical mathematical model. -/
theorem real_cast_eval (x : Op .real shape) (s : BlockState) :
    evalOp (dtype := .real) (.castFloat .real .real x) s = evalOp x s := by
  change evalOp (.castFloat .real .real x) s = evalOp x s
  rw [evalOp_castFloat]
  cases hx : evalOp x s <;> simp [hx, FloatDType.cast]

/-- This is an exact-real theorem. It is deliberately absent from FP rules. -/
theorem powi_square_eval (x : Op .real [B]) (s : BlockState) :
    evalOp (.libdevicePowi .scalarR x (.constInt 2)) s =
      evalOp (.mul .real (.consSame .nil) x x) s := by
  rw [evalOp_libdevicePowi, evalOp_mul]
  have hn : evalOp (.constInt 2) s = some (Tile.scalar 2) := by simp only [evalOp]
  rw [hn]
  cases hx : evalOp x s with
  | none => rfl
  | some v =>
      apply congrArg some
      ext i
      simp only [Tile.libdevicePowi_data, Tile.bop_data, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Tile.scalar_data, NumericDType.mul]
      exact WithBot.realPowi_two (v.data i)

theorem variance_assignment_flat (A : FlatAlloc) (B : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts
      [.assign .real [B] "var" (.add .real (.consSame .nil)
        (.ref .real [B] "var")
        (.libdevicePowi .scalarR (.ref .real [B] "x") (.constInt 2)))]) s =
      stepStmts (A.flattenStmts
      [.assign .real [B] "var" (.add .real (.consSame .nil)
        (.ref .real [B] "var")
        (.mul .real (.consSame .nil)
          (.castFloat .real .real (.ref .real [B] "x"))
          (.castFloat .real .real (.ref .real [B] "x"))))]) s := by
  simp only [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
    stepStmts, stepStmt, evalOp_add, powi_square_eval, evalOp_mul]
  simp only [real_cast_eval, FloatDType.toTileDType]

theorem variance_flat (A : FlatAlloc) (X : RegionName) (sxk N B : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (powiVarBody X sxk N B)) s =
      stepStmts (A.flattenStmts (varBody X sxk N B)) s := by
  have hbody : varBody X sxk N B = (varBody X sxk N B).take 3 ++
      [.assign .real [B] "var" (.add .real (.consSame .nil)
        (.ref .real [B] "var")
        (.mul .real (.consSame .nil)
          (.castFloat .real .real (.ref .real [B] "x"))
          (.castFloat .real .real (.ref .real [B] "x"))))] := rfl
  conv_rhs => rw [hbody]
  simp only [powiVarBody, FlatAlloc.flattenStmts_append, ProgramRewrite.append,
    variance_assignment_flat]

theorem prefix_flat (A : FlatAlloc) (v : Bool) (sxb sxm : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix v sxb sxm)) s =
      stepStmts (A.flattenStmts (indexPrefix Bool.false sxb sxm)) s := by
  cases v
  · rfl
  · simp [indexPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
      stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

/-- Every source variant has exactly the retained real program's full state
transition, through both complete tiled loops. The only numerical bridge is
integer exponent two equals square on ℝ; the FP theorem never uses it. -/
theorem typed_flat_exec (A : FlatAlloc) (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) (s : BlockState) :
    exec (A.flattenKernel
      (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel
        (rmsnorm_triton X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel) s := by
  rw [typed_projection]
  have hloop : ∀ t : BlockState,
      stepStmts (A.flattenStmts [.forRange "block_n_start_idx" 0 N B (powiVarBody X sxk N B)]) t =
      stepStmts (A.flattenStmts [.forRange "block_n_start_idx" 0 N B (varBody X sxk N B)]) t := by
    intro t
    simp only [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, stepStmts, stepStmt]
    rw [ProgramRewrite.range_congr _ _ _ _ _ _ (variance_flat A X sxk N B)]
  have hp : prefixStmts X sxb sxm B =
      indexPrefix Bool.false sxb sxm ++ (prefixStmts X sxb sxm B).drop 3 := rfl
  unfold exec FlatAlloc.flattenKernel
  rw [math_body, Full.Final.body_decomp]
  conv_rhs => rw [hp]
  simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, hloop, prefix_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormTriton
