import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatData
import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatSyntax

/-! Typed wrapper and classifier relocation. The data given to every cast,
load callback and classifier is identical. Optional callbacks are neither
replaced nor assumed present. Unsupported constructors retain `none`. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)
variable (s : State α)

def ComputeSame {d : ComputeDType} {sh : TileShape} (e : ComputeOp d sh) : Prop :=
  evalComputeOp M (flattenComputeOp A e) (flattenState A background flatUndef s) =
    evalComputeOp M e s

def ComputeCommutes {d : ComputeDType} {sh : TileShape} (e : ComputeOp d sh) : Prop :=
  evalComputeOp M (flattenComputeOp A e) (flattenState A background flatUndef s) =
    (evalComputeOp M e s).map (trValues A)

def ExprSame {d : AlgDType} {sh : TileShape} (e : ComputeExpr d sh) : Prop :=
  evalExpr M (flattenExpr A e) (flattenState A background flatUndef s) = evalExpr M e s

def ExprCommutes {d : AlgDType} {sh : TileShape} (e : ComputeExpr d sh) : Prop :=
  evalExpr M (flattenExpr A e) (flattenState A background flatUndef s) =
    (evalExpr M e s).map (trValues A)

theorem computeSame_iff_commutes {d : ComputeDType} {sh : TileShape} (e : ComputeOp d sh) :
    ComputeSame A M background flatUndef s e ↔ ComputeCommutes A M background flatUndef s e := by
  have hp : d.eraseDType ≠ .ptr := by cases d <;> decide
  have hb : d.eraseDType ≠ .blockPtr := by cases d <;> decide
  unfold ComputeSame ComputeCommutes
  rw [map_trValues_data A _ hp hb]

theorem exprSame_iff_commutes {d : AlgDType} {sh : TileShape} (e : ComputeExpr d sh)
    (hp : d ≠ .ptr) (hb : d ≠ .blockPtr) :
    ExprSame A M background flatUndef s e ↔ ExprCommutes A M background flatUndef s e := by
  unfold ExprSame ExprCommutes
  rw [map_trValues_data A _ hp hb]

theorem compute_alg (d : ComputeDType) {sh : TileShape} (e : Op d.eraseDType sh)
    (he : OpSame A M background flatUndef s (some d) e) :
    ComputeSame A M background flatUndef s (.alg d e) := by
  unfold ComputeSame
  rw [flattenComputeOp]
  change evalOp M (some d) (A.flattenOp e) (flattenState A background flatUndef s) =
    evalOp M (some d) e s
  exact he

theorem compute_const (d : ComputeDType) (v : ComputeCarrier d) :
    ComputeSame A M background flatUndef s (.const v) := by
  cases d <;> simp only [ComputeSame, flattenComputeOp, evalComputeOp]

theorem compute_full {d : ComputeDType} (sh : TileShape) (e : ComputeOp d [])
    (he : ComputeSame A M background flatUndef s e) :
    ComputeSame A M background flatUndef s (.full sh e) := by
  unfold ComputeSame at *
  rw [flattenComputeOp]
  change ((evalComputeOp M (flattenComputeOp A e) (flattenState A background flatUndef s)).bind fun v =>
    some (fun (_ : TileIndex sh) => v PUnit.unit)) =
    ((evalComputeOp M e s).bind fun v => some (fun (_ : TileIndex sh) => v PUnit.unit))
  rw [he]

/-- The fp32 load callback is applied on both sides after the same `.real`
storage read. In particular, this does not identify that callback with id. -/
theorem compute_load_fp32 {sh : TileShape} (mem : MemAccess .real sh)
    (mask : MaskOpt .real sh) (padding : PaddingOption)
    (he : OpSame A M background flatUndef s (some .fp32) (.load .real mem mask padding)) :
    ComputeSame A M background flatUndef s (.load .fp32 mem mask padding) := by
  have he' : evalOp M (some .fp32)
      (.load .real (A.flattenAccess mem) (A.flattenMask mask) padding)
      (flattenState A background flatUndef s) = evalOp M (some .fp32) (.load .real mem mask padding) s := by
    unfold OpSame at he
    exact (congrArg (fun op : Op .real sh =>
      evalOp M (some .fp32) op (flattenState A background flatUndef s))
      (FlatAlloc.flattenOp_load A mem mask padding)).symm.trans he
  unfold ComputeSame
  rw [flattenComputeOp]
  change ((evalOp M (some .fp32)
      (.load .real (A.flattenAccess mem) (A.flattenMask mask) padding)
      (flattenState A background flatUndef s)).bind fun v => some (fun i => M.fp32Load (v i))) =
    ((evalOp M (some .fp32) (.load .real mem mask padding) s).bind fun v =>
      some (fun i => M.fp32Load (v i)))
  rw [he']

/-- Direct ComputeOp fp64 loads are unsupported in the current evaluator.
The source's supported fp64 path uses an explicitly precise ordinary load. -/
theorem compute_load_fp64 {sh : TileShape} (mem : MemAccess .real sh)
    (mask : MaskOpt .real sh) (padding : PaddingOption) :
    ComputeSame A M background flatUndef s (.load .fp64 mem mask padding) := by
  simp only [ComputeSame, flattenComputeOp, evalComputeOp]

theorem compute_bitcast (src dst : ComputeDType) {sh : TileShape}
    (width : src.width = dst.width) (e : ComputeOp src sh)
    (he : ComputeSame A M background flatUndef s e) :
    ComputeSame A M background flatUndef s (.bitcast src dst width e) := by
  unfold ComputeSame at *
  cases src <;> cases dst <;>
    simp only [flattenComputeOp, evalComputeOp, he]

theorem expr_alg {d : AlgDType} {sh : TileShape} (e : Op d sh)
    (he : OpSame A M background flatUndef s none e) :
    ExprSame A M background flatUndef s (.alg e) := by
  unfold ExprSame
  rw [flattenExpr]
  change evalOp M none (A.flattenOp e) (flattenState A background flatUndef s) = evalOp M none e s
  exact he

theorem expr_compute {d : ComputeDType} {sh : TileShape} (e : ComputeOp d sh)
    (he : ComputeSame A M background flatUndef s e) :
    ExprSame A M background flatUndef s (.compute e) := by
  unfold ExprSame
  rw [flattenExpr]
  change evalComputeOp M (flattenComputeOp A e) (flattenState A background flatUndef s) =
    evalComputeOp M e s
  exact he

theorem expr_opaque {d : AlgDType} {sh : TileShape} (name : String) :
    ExprSame A M background flatUndef s (ComputeExpr.opaque (dtype := d) (shape := sh) name) := by
  simp only [ExprSame, flattenExpr, evalExpr]

theorem expr_llrint {sh : TileShape} (e : ComputeExpr .real sh) :
    ExprSame A M background flatUndef s (.llrint e) := by
  simp only [ExprSame, flattenExpr, evalExpr]

/-- Arbitrary callback, arbitrary precision, arbitrary data, including an
empty tile. This equality preserves child failure and missing callbacks. -/
theorem expr_isFinite {sh : TileShape} (precision : FinitePrecision)
    (e : ComputeExpr .real sh) (he : ExprSame A M background flatUndef s e) :
    ExprSame A M background flatUndef s (.isFinite precision e) := by
  unfold ExprSame at *
  rw [flattenExpr]
  change ((evalExpr M (flattenExpr A e) (flattenState A background flatUndef s)).bind fun values =>
    (M.isFinite precision).bind fun classify => some (fun i => classify (values i))) =
    ((evalExpr M e s).bind fun values =>
      (M.isFinite precision).bind fun classify => some (fun i => classify (values i)))
  rw [he]

/-- Missing classification support cannot become success after relocation. -/
theorem isFinite_missing {sh : TileShape} (precision : FinitePrecision)
    (e : ComputeExpr .real sh) (hmissing : M.isFinite precision = none) :
    evalExpr M (.isFinite precision e) s = none := by
  change ((evalExpr M e s).bind fun values =>
    (M.isFinite precision).bind fun classify => some (fun i => classify (values i))) = none
  cases hchild : evalExpr M e s <;>
    simp only [bind, Option.bind, hmissing]

theorem isFinite_missing_flat {sh : TileShape} (precision : FinitePrecision)
    (e : ComputeExpr .real sh) (hmissing : M.isFinite precision = none) :
    evalExpr M (flattenExpr A (.isFinite precision e))
        (flattenState A background flatUndef s) = none ∧
      evalExpr M (.isFinite precision e) s = none := by
  constructor
  · exact isFinite_missing M (flattenState A background flatUndef s)
      precision (flattenExpr A e) hmissing
  · exact isFinite_missing M s precision e hmissing

#axiomsClean computeSame_iff_commutes
#axiomsClean exprSame_iff_commutes
#axiomsClean compute_alg
#axiomsClean compute_const
#axiomsClean compute_full
#axiomsClean compute_load_fp32
#axiomsClean compute_load_fp64
#axiomsClean compute_bitcast
#axiomsClean expr_alg
#axiomsClean expr_compute
#axiomsClean expr_opaque
#axiomsClean expr_llrint
#axiomsClean expr_isFinite
#axiomsClean isFinite_missing
#axiomsClean isFinite_missing_flat

end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
