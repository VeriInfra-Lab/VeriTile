import bench.tritonbench_g.layer_norm_fwd.Proofs.Source
import bench.tritonbench_g.layer_norm_fwd.Proofs.Domain
import VeriTile.Triton.Float.Erasure

/-! Source fidelity gate: the nested seed expression must load the signed
int64 memory channel before modulo 2^32. Source admission alone cannot
establish this ABI fact. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.SeedSource
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def firstAssignment (k : Kernel) : Option Stmt := do
  let .ifThen _ body ← k.body[11]? | none
  body[0]?

def keepAssignment (r : Regions) (p : ℝ) (block : Nat) : Stmt :=
  .assign .bool [block] "keep_mask"
    (.gt .real .scalarR
      (.rand Domain.randomProfile
        (.castNatToInt (Op.intToUnsigned 32
          (.load .int (.region r.seeds.cast (.ref .nat [] "row")) .none)))
        (.castNatToInt (.ref .nat [block] "cols"))) (.const p))

theorem original_signed_seed (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    firstAssignment (original f r stride M N eps p block hBlock).eraseDType.toAlgKernel =
      some (keepAssignment r p block) := by
  have hproj : (original f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (original f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, hproj, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [original, firstAssignment, keepAssignment, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, Random.i32ArangeSize, Region.cast,
    List.getElem?_cons_succ, List.getElem?_cons_zero]
  rfl

def secondAssignment (k : Kernel) : Option Stmt := do
  let .ifThen _ inputBody ← k.body[12]? | none
  let .ifThen _ dropoutBody ← inputBody[2]? | none
  dropoutBody[0]?

def secondKeepAssignment (r : Regions) (M : Nat) (p : ℝ) (block : Nat) : Stmt :=
  .assign .bool [block] "keep_mask"
    (.gt .real .scalarR
      (.rand Domain.randomProfile
        (.castNatToInt (Op.intToUnsigned 32
          (.load .int (.region r.seeds.cast
            (.add .nat .nil (.constNat M) (.ref .nat [] "row"))) .none)))
        (.castNatToInt (.ref .nat [block] "cols"))) (.const p))

theorem original_second_signed_seed (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    secondAssignment (original f r stride M N eps p block hBlock).eraseDType.toAlgKernel =
      some (secondKeepAssignment r M p block) := by
  have hproj : (original f r stride M N eps p block hBlock).toAlgorithm? =
      .ok (original f r stride M N eps p block hBlock).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, hproj, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [original, secondAssignment, secondKeepAssignment, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, Random.i32ArangeSize, Region.cast,
    List.getElem?_cons_succ, List.getElem?_cons_zero]
  rfl

end VeriTile.Bench.TritonBenchG.LayerNormFwd.SeedSource
