import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Execution

/-! The source rowscale phase. Both the first and second input use this
operation; the second uses scalar index `M + row`. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def scalePhase (enabled : Bool) (r : RegionName) (block : Nat) : List Stmt :=
  [.ifThen (.constBool enabled)
    [.assign .real [] "rowscale" (.load .real (.region r (.ref .nat [] "row")) .none),
     .assign .real [block] "x" (.mul .real .scalarR
       (.ref .real [block] "x") (.ref .real [] "rowscale"))]]

theorem source_scalePhase (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    ((source f r stride M N eps p block hBlock).body.drop 10).take 1 =
      scalePhase f.rowscale r.rowscale block := by
  have hproj : (original f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (original f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [source, ComputeKernel.eraseDType, hproj, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [original, scalePhase, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
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

def realTile {block : Nat} (values : Fin block → ℝ) : Tile .real [block] :=
  ⟨fun j => some (values j.1)⟩

def scaleState (enabled : Bool) (r : RegionName) (row block : Nat)
    (values : Fin block → ℝ) (s : BlockState) : BlockState :=
  if enabled then
    let factor := s.readMem r row
    let t := s.setReg "rowscale" .real [] (Tile.scalar (some factor))
    t.setReg "x" .real [block] (realTile (fun j => values j * factor))
  else s

theorem scalePhase_exec (enabled : Bool) (r : RegionName) (row block : Nat)
    (values : Fin block → ℝ) (s : BlockState)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hx : s.regs .real [block] "x" = some (realTile values)) :
    stepStmts (scalePhase enabled r block) s =
      some (scaleState enabled r row block values s) := by
  cases enabled
  · simp [scalePhase, scaleState, stepStmts, stepStmt, evalOp.eq_def]
  · simp [scalePhase, scaleState, stepStmts, stepStmt, evalOp.eq_def,
      hrow, hx, realTile, Tile.bop, NumericDType.mul]

theorem scaleState_mem (enabled : Bool) (r : RegionName) (row block : Nat)
    (values : Fin block → ℝ) (s : BlockState) :
    (scaleState enabled r row block values s).mem = s.mem := by
  cases enabled <;> rfl

theorem scaleState_x (enabled : Bool) (r : RegionName) (row block : Nat)
    (values : Fin block → ℝ) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (realTile values)) :
    (scaleState enabled r row block values s).regs .real [block] "x" =
      some (realTile (fun j => if enabled then values j * s.readMem r row else values j)) := by
  cases enabled <;> simp [scaleState, hx]

theorem scaleState_row (enabled : Bool) (r : RegionName) (row block : Nat)
    (values : Fin block → ℝ) (s : BlockState) :
    (scaleState enabled r row block values s).regs .nat [] "row" = s.regs .nat [] "row" := by
  cases enabled <;> simp [scaleState]

theorem scaleState_cols (enabled : Bool) (r : RegionName) (row block : Nat)
    (values : Fin block → ℝ) (s : BlockState) :
    (scaleState enabled r row block values s).regs .nat [block] "cols" =
      s.regs .nat [block] "cols" := by
  cases enabled <;> simp [scaleState]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
