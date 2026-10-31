import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Residual

/-! The complete remaining source subtree. Conditional expressions use the
source's lazy scalar `ite`, and masked coefficient loads have no `other` value;
their inactive lanes retain the operational undef semantics. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def maskStmt (N block : Nat) : Stmt :=
  .assign .bool [block] "mask"
    (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N))

def coefficientOp (region : RegionName) (block : Nat) : Op .real [block] :=
  .load .real (.region region (.ref .nat [block] "cols"))
    (.mask (.ref .bool [block] "mask"))

def loadPhase (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (block : Nat) : List Stmt :=
  [.assign .real [block] weightName (coefficientOp weightRegion block),
   .ifThen (.constBool bias) [.assign .real [block] biasName (coefficientOp biasRegion block)]]

def normalizedOp (rms : Bool) (block : Nat) : Op .real [block] :=
  .ite (.boolNot (.constBool rms))
    (.mul .real .scalarR
      (.sub .real .scalarR (.ref .real [block] "x") (.ref .real [] "mean"))
      (.ref .real [] "rstd"))
    (.mul .real .scalarR (.ref .real [block] "x") (.ref .real [] "rstd"))

def valueOp (bias : Bool) (weightName biasName : RegName) (block : Nat) : Op .real [block] :=
  let weighted := .mul .real (.consSame .nil)
    (.ref .real [block] "x_hat") (.ref .real [block] weightName)
  .ite (.constBool bias)
    (.add .real (.consSame .nil) weighted (.ref .real [block] biasName)) weighted

def storeStmt (pointer name : RegName) (block : Nat) : Stmt :=
  .store .real [block]
    (.ptr (.ptrAdd .scalarL (.ref .ptr [] pointer) (.ref .nat [block] "cols")))
    (.ref .real [block] name) (.mask (.ref .bool [block] "mask"))

def outputPhase (bias : Bool) (weightName biasName outputName pointer : RegName)
    (block : Nat) : List Stmt :=
  [.assign .real [block] outputName (valueOp bias weightName biasName block),
   storeStmt pointer outputName block]

def secondStmt (f : Flags) (r : Regions) (block : Nat) : Stmt :=
  .ifThen (.constBool f.w1)
    (loadPhase f.b1 r.weight1 r.bias1 "w1" "b1" block ++
      outputPhase f.b1 "w1" "b1" "y1" "Y1" block)

def phase (f : Flags) (r : Regions) (N block : Nat) : List Stmt :=
  [maskStmt N block] ++ loadPhase f.bias r.weight r.bias "w" "b" block ++
    [.assign .real [block] "x_hat" (normalizedOp f.rms block)] ++
    outputPhase f.bias "w" "b" "y" "Y" block ++ [secondStmt f r block]

theorem source_phase (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    (Execution.source f r stride M N eps p block hBlock).body.drop 18 =
      phase f r N block := by
  have hproj : (original f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (original f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [Execution.source, ComputeKernel.eraseDType, hproj,
    ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [original, phase, maskStmt, loadPhase, coefficientOp, normalizedOp,
    outputPhase, valueOp, storeStmt, secondStmt,
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

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
