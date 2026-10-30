import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_NormalizationExecution

/-! Exact source statements between the first stream and normalization.
Both the second seed and rowscale use M + row, while X1 uses its own row stride. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.InputSuffix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def rowLoadOp (pointer : RegName) (N block : Nat) : Op .real [block] :=
  .load .real (.ptr (.ptrAdd .scalarL (.ref .ptr [] pointer) (.ref .nat [block] "cols")))
    (.maskOther (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N))
      (.broadcast (.const 0.0) [block]))

def secondRowOp (M : Nat) : Op .nat [] :=
  .add .nat .nil (.constNat M) (.ref .nat [] "row")

def secondRandomOp (seeds : RegionName) (M block : Nat) : Op .real [block] :=
  .rand Domain.randomProfile
    (.castNatToInt (Op.intToUnsigned 32
      (.load .int (.region seeds.cast (secondRowOp M)) .none)))
    (.castNatToInt (.ref .nat [block] "cols"))

def secondKeepOp (seeds : RegionName) (M block : Nat) (p : ℝ) : Op .bool [block] :=
  .gt .real .scalarR (secondRandomOp seeds M block) (.const p)

def dropoutValueOp (block : Nat) (p : ℝ) : Op .real [block] :=
  .where (.ref .bool [block] "keep_mask")
    (.div .real .scalarR (.ref .real [block] "x1")
      (.sub .real .nil (.const 1.0) (.const p)))
    (.broadcast (.const 0.0) [block])

def maskStmt (mask : RegionName) (M N block : Nat) : Stmt :=
  .store .bool [block]
    (.region mask.cast (.add .nat .scalarL
      (.mul .nat .nil (secondRowOp M) (.constNat N)) (.ref .nat [block] "cols")))
    (.ref .bool [block] "keep_mask")
    (.mask (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N)))

def scalePhase (enabled : Bool) (rowscale : RegionName) (M block : Nat) : List Stmt :=
  [.ifThen (.constBool enabled)
    [.assign .real [] "rowscale" (.load .real (.region rowscale (secondRowOp M)) .none),
     .assign .real [block] "x1" (.mul .real .scalarR
       (.ref .real [block] "x1") (.ref .real [] "rowscale"))]]

def dropoutPhase (enabled storeMask : Bool) (seeds mask : RegionName)
    (M N block : Nat) (p : ℝ) : List Stmt :=
  [.ifThen (.constBool enabled)
    [.assign .bool [block] "keep_mask" (secondKeepOp seeds M block p),
     .assign .real [block] "x1" (dropoutValueOp block p),
     .ifThen (.constBool storeMask) [maskStmt mask M N block]]]

def addStmt (name : RegName) (block : Nat) : Stmt :=
  .assign .real [block] "x" (.add .real (.consSame .nil)
    (.ref .real [block] "x") (.ref .real [block] name))

def secondBody (f : Flags) (r : Regions) (M N block : Nat) (p : ℝ) : List Stmt :=
  [.assign .real [block] "x1" (rowLoadOp "X1" N block)] ++
    scalePhase f.rowscale r.rowscale M block ++
    dropoutPhase f.dropout f.dropoutMask r.seeds r.dropoutMask M N block p ++
    [addStmt "x1" block]

def residualStmt (enabled : Bool) (N block : Nat) : Stmt :=
  .ifThen (.constBool enabled)
    [.assign .real [block] "residual" (rowLoadOp "RESIDUAL" N block), addStmt "residual" block]

def residualOutStmt (enabled : Bool) (N block : Nat) : Stmt :=
  .ifThen (.constBool enabled)
    [.store .real [block]
      (.ptr (.ptrAdd .scalarL (.ref .ptr [] "RESIDUAL_OUT") (.ref .nat [block] "cols")))
      (.ref .real [block] "x")
      (.mask (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N)))]

def phase (f : Flags) (r : Regions) (M N block : Nat) (p : ℝ) : List Stmt :=
  [.ifThen (.constBool f.x1) (secondBody f r M N block p),
   residualStmt f.residual N block, residualOutStmt f.residualOut N block]

theorem source_phase (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    ((Execution.source f r stride M N eps p block hBlock).body.drop 12).take 3 =
      phase f r M N block p := by
  have hproj : (original f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (original f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [Execution.source, ComputeKernel.eraseDType, hproj,
    ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [original, phase, secondBody, rowLoadOp, scalePhase, secondRowOp,
    dropoutPhase, secondKeepOp, secondRandomOp, dropoutValueOp, maskStmt,
    addStmt, residualStmt, residualOutStmt,
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

def rowValues (region : RegionName) (start N block : Nat) (s : BlockState) : Fin block → ℝ :=
  fun j => if j.val < N then s.readMem region (start+j.val) else 0

theorem rowLoadOp_eval (pointer : RegName) (region : RegionName) (start N block : Nat)
    (s : BlockState)
    (hptr : s.regs .ptr [] pointer = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val))) :
    evalOp (rowLoadOp pointer N block) s =
      some (Execution.realTile (rowValues region start N block s)) := by
  simp only [rowLoadOp, evalOp, hptr, hcols, bind, Option.bind, pure]
  congr 1
  ext j
  by_cases hj : j.1.val < N
  all_goals simp [Tile.ptrAdd, Tile.cop, ComparableDType.lt,
    Execution.realTile, rowValues, hj] <;> norm_num

theorem secondRandomOp_eval (seeds : RegionName) (M row block : Nat) (s : BlockState)
    (hBlock : Random.ArangeSizeDomain block)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val))) :
    evalOp (secondRandomOp seeds M block) s =
      some ⟨fun j => some (Random.randReal
        (Formula.seedU32 (s.readMemValue .int seeds (M+row))) (j.1.val : Int) .bits32 7)⟩ := by
  unfold secondRandomOp
  apply Domain.eval_rand _ hBlock
  · apply Domain.eval_seed_cast
    simp [secondRowOp, evalOp, hrow, Tile.bop, NumericDType.add, Region.cast]
  · simp [evalOp, hcols]

theorem secondKeepOp_eval (seeds : RegionName) (M row block : Nat) (p : ℝ) (s : BlockState)
    (hBlock : Random.ArangeSizeDomain block)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val))) :
    evalOp (secondKeepOp seeds M block p) s =
      some (Execution.keepTile (s.readMemValue .int seeds (M+row)) p block) := by
  unfold secondKeepOp
  rw [evalOp_gt, secondRandomOp_eval seeds M row block s hBlock hrow hcols]
  simp only [evalOp_const, bind, Option.bind, pure]
  congr 1
  ext j
  simp only [Execution.keepTile, Tile.cop, ComparableDType.gt, Formula.keep,
    Tile.scalar, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    WithBot.some_eq_coe, WithBot.coe_lt_coe]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.InputSuffix
