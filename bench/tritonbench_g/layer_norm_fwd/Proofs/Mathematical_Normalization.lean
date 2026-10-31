import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_FirstStream
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Reduction

/-! Exact source normalization phase, after the combined input has been
established by the input phases. The phase hypotheses are not a full-kernel
contract: the second input and residual must still establish them. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Normalization
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def meanStmt (N block : Nat) : Stmt :=
  .assign .real [] "mean" (.div .real .nil
    (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [block] "x")) (.const (N : ℝ)))

def scalarStore (region : RegionName) (name : RegName) : Stmt :=
  .store .real [] (.region region (.ref .nat [] "row")) (.ref .real [] name) .none

def centerStmt (rms : Bool) (N block : Nat) : Stmt :=
  .assign .real [block] "xbar"
    (.where (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N))
      (if rms then .ref .real [block] "x" else
        .sub .real .scalarR (.ref .real [block] "x") (.ref .real [] "mean"))
      (.broadcast (.const 0.0) [block]))

def varianceStmt (N block : Nat) : Stmt :=
  .assign .real [] "var" (.div .real .nil
    (.reduceSum ⟨0, by simp⟩ Bool.false
      (.mul .real (.consSame .nil) (.ref .real [block] "xbar") (.ref .real [block] "xbar")))
    (.const (N : ℝ)))

def centerVariancePhase (rms : Bool) (N block : Nat) : List Stmt :=
  [centerStmt rms N block, varianceStmt N block]

def meanPhase (mean : RegionName) (N block : Nat) : List Stmt :=
  [meanStmt N block, scalarStore mean "mean"]

def statisticsStmt (rms : Bool) (mean : RegionName) (N block : Nat) : Stmt :=
  .ifThenElse (.boolNot (.constBool rms))
    (meanPhase mean N block ++ centerVariancePhase Bool.false N block)
    (centerVariancePhase Bool.true N block)

def rstdStmt (eps : ℝ) : Stmt :=
  .assign .real [] "rstd" (.div .real .nil (.const 1)
    (.sqrt (.add .real .nil (.ref .real [] "var") (.const eps))))

def phase (f : Flags) (r : Regions) (N block : Nat) (eps : ℝ) : List Stmt :=
  [statisticsStmt f.rms r.mean N block, rstdStmt eps, scalarStore r.rstd "rstd"]

theorem source_phase (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    ((Execution.source f r stride M N eps p block hBlock).body.drop 15).take 3 =
      phase f r N block eps := by
  have hproj : (original f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (original f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [Execution.source, ComputeKernel.eraseDType, hproj,
    ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [original, phase, statisticsStmt, meanPhase, meanStmt, scalarStore,
    centerVariancePhase, centerStmt, varianceStmt, rstdStmt,
    ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal,VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
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

def centerVarianceState (f : Flags) (a : Inputs) (M N block : Nat) (p : ℝ)
    (row : Nat) (s : BlockState) : BlockState :=
  (s.setReg "xbar" .real [block] (Reduction.centeredTile f a M N block p row)).setReg
    "var" .real [] (Tile.scalar (some (Formula.variance f a M N p row)))

theorem centerVariance_exec (f : Flags) (a : Inputs) (M N block : Nat)
    (p : ℝ) (row : Nat) (hN : N ≤ block) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (Reduction.combinedTile f a M N block p row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmean : f.rms = Bool.false → s.regs .real [] "mean" =
      some (Tile.scalar (some (Formula.mean f a M N p row)))) :
    stepStmts (centerVariancePhase f.rms N block) s =
      some (centerVarianceState f a M N block p row s) := by
  have hc := stepStmt_assign_eq_some (name := "xbar")
    (Reduction.centered_op f a M N block p row s hx hcols hmean)
  have hv := stepStmt_assign_eq_some (name := "var")
    (Reduction.variance_op f a M N block p row hN
      (s.setReg "xbar" .real [block] (Reduction.centeredTile f a M N block p row))
      (by simp))
  unfold centerVariancePhase centerStmt varianceStmt
  rw [stepStmts.cons_some hc, stepStmts.cons_some hv]
  rw [stepStmts.nil]
  rfl

def meanState (f : Flags) (a : Inputs) (mean : RegionName) (M N : Nat)
    (p : ℝ) (row : Nat) (s : BlockState) : BlockState :=
  Memory.writeCell (s.setReg "mean" .real []
    (Tile.scalar (some (Formula.mean f a M N p row)))) mean row
    (MemCell.real (Formula.mean f a M N p row))

theorem mean_exec (f : Flags) (a : Inputs) (mean : RegionName) (M N block : Nat)
    (p : ℝ) (row : Nat) (hN : N ≤ block) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (Reduction.combinedTile f a M N block p row))
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row)) :
    stepStmts (meanPhase mean N block) s = some (meanState f a mean M N p row s) := by
  have hm := stepStmt_assign_eq_some (name := "mean")
    (Reduction.mean_op f a M N block p row hN s hx)
  have hs := Memory.scalar_store mean "mean" row (Formula.mean f a M N p row)
    (s.setReg "mean" .real [] (Tile.scalar (some (Formula.mean f a M N p row))))
    (by simpa using hrow) (by simp)
  unfold meanPhase meanStmt scalarStore
  rw [stepStmts.cons_some hm, stepStmts.cons_some hs]
  rw [stepStmts.nil]
  rfl

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Normalization
