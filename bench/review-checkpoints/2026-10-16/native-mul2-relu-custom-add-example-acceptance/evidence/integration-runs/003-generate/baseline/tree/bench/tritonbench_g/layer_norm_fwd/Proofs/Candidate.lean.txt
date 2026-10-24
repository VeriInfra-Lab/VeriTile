import bench.tritonbench_g.layer_norm_fwd.Proofs.Real

/-! Independent real execution of the candidate's changed source prefix,
followed by the common complete source suffix. This proof does not import or
use floating-point equivalence. The mathematical contract is the same initial
snapshot specification for every output and every unchanged tagged cell. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.CandidateExecution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

def source (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : Kernel :=
  (candidate f r stride M N eps p block hBlock).eraseDType.toAlgKernel

def baseStmt (r : RegionName) (stride : Nat) : Stmt :=
  .assign .ptr [] "X" (.ptrAdd .nil (.ptrBase r)
    (.mul .nat .nil (.constNat stride) (.ref .nat [] "row")))

def loadPrefix (f : Flags) (r : Regions) (stride : Strides) (N block : Nat) : List Stmt :=
  [.assign .nat [] "row" (.programId 0), baseStmt r.x stride.x] ++
    (Execution.loadPrefix f r stride N block).drop 2

theorem source_loadPrefix (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    (source f r stride M N eps p block hBlock).body.take 10 = loadPrefix f r stride N block := by
  have hproj : (candidate f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (candidate f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [source, ComputeKernel.eraseDType, hproj, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [candidate, loadPrefix, baseStmt, Execution.loadPrefix, Execution.baseStmt,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, Random.i32ArangeSize, Region.cast,
    List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero]
  rfl

theorem loadPrefix_exec (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) :
    stepStmts (loadPrefix f r stride N block) u =
      some (Execution.loadState f r stride N block u) := by
  rcases f with ⟨rms, residual, residualOut, bias, dropout, dropoutMask, rowscale, x1, w1, b1⟩
  cases residual <;> cases residualOut <;> cases x1 <;> cases w1
  all_goals simp [loadPrefix, baseStmt, Execution.loadPrefix, Execution.baseStmt,
    Execution.loadState, Execution.ptrState, Execution.loaded,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
    NumericDType.mul, ComparableDType.lt, Nat.mul_comm]

theorem source_initial_load (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) :
    stepStmts ((source f r stride M N eps p block hBlock).body.take 10) u =
      some (Execution.loadState f r stride N block u) := by
  rw [source_loadPrefix, loadPrefix_exec]

theorem common_suffix (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    (source f r stride M N eps p block hBlock).body.drop 10 =
      (Execution.source f r stride M N eps p block hBlock).body.drop 10 := by
  have hc : (candidate f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (candidate f r stride M N eps p block hBlock).toAlgKernel := rfl
  have ho : (original f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (original f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [source, Execution.source, ComputeKernel.eraseDType, hc, ho,
    ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [candidate, original, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, Random.i32ArangeSize, Region.cast,
    List.drop_succ_cons, List.drop_zero]

theorem common_suffix_exec (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    stepStmts ((source f r stride M N eps p block hBlock).body.drop 10)
      (Execution.loadState f r stride N block u) =
      some (RealExecution.state f r stride M N block eps p u) := by
  rw [common_suffix, ← Execution.source_after_load]
  exact RealExecution.source_exec f r stride M N eps p block hBlock u hN hsep

theorem source_exec (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    exec (source f r stride M N eps p block hBlock) u =
      some (RealExecution.state f r stride M N block eps p u) := by
  unfold exec
  conv_lhs => rw [← List.take_append_drop 10 (source f r stride M N eps p block hBlock).body]
  rw [ProgramRewrite.append, source_initial_load, Option.bind_some]
  exact common_suffix_exec f r stride M N eps p block hBlock u hN hsep

theorem source_correct (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    ∃ final, exec (source f r stride M N eps p block hBlock) u = some final ∧
      Contract.Post f r stride M N eps p u final :=
  ⟨RealExecution.state f r stride M N block eps p u,
    source_exec f r stride M N eps p block hBlock u hN hsep,
    RealExecution.source_post f r stride M N block eps p u hN hsep⟩

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.CandidateExecution
