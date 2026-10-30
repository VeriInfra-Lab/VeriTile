import bench.tritonbench_g.uniform_sampling.Kernels
import VeriTile.Triton.Float.Erasure

/-! Checked projection of the complete expanded source. Projection success
is explicit: no error-default kernel can stand in for an unsupported source.
Floating erasure retains all integer operations, ten Philox rounds, all four
uniform decoders, and the four independent masked stores.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

theorem projection_ok (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).toAlgorithm? =
      .ok (source storage swapped y N seed offset lower upper B hB).toAlgKernel := by
  cases storage <;> cases swapped <;> rfl

/-- Three direct materializations plus all twenty original source statements.
The tuple statement contains its five captures and all four result writes. -/
theorem full_statement_count (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).surfaceBody.length = 23 := by
  cases storage <;> cases swapped <;> rfl

def reference (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : Kernel :=
  (source .fp64 swapped y N seed offset lower upper B hB).toAlgKernel.eraseDType

def erasedBody (body : List ComputeStmt) : Except EraseDTypeError (List Stmt) := do
  return Stmt.eraseDTypeList (← ComputeStmt.listToAlgorithm? body)

theorem eraseList_append (before after : List Stmt) :
    Stmt.eraseDTypeList (before ++ after) =
      Stmt.eraseDTypeList before ++ Stmt.eraseDTypeList after := by
  induction before with
  | nil => simp only [List.nil_append, Stmt.eraseDTypeList]
  | cons head rest ih => simp only [List.cons_append, Stmt.eraseDTypeList, ih]

theorem erasedBody_append (before after : List ComputeStmt) :
    erasedBody (before ++ after) = (do
      let a ← erasedBody before
      let b ← erasedBody after
      return a ++ b) := by
  simp only [erasedBody, ComputeStmt.listToAlgorithm?_append]
  cases ComputeStmt.listToAlgorithm? before <;>
    cases ComputeStmt.listToAlgorithm? after <;>
    simp only [bind, pure, Except.bind, Except.pure, eraseList_append]

theorem erasedBody_eq_of_parts (a b : List ComputeStmt) (n : Nat)
    (hp : a.take n = b.take n) (hs : erasedBody (a.drop n) = erasedBody (b.drop n)) :
    erasedBody a = erasedBody b := by
  conv_lhs => rw [← List.take_append_drop n a]
  conv_rhs => rw [← List.take_append_drop n b]
  rw [erasedBody_append, erasedBody_append, hp, hs]

theorem erased_projection_mk (inputs outputs : List RegionName) (body : List ComputeStmt) :
    (ComputeKernel.mk inputs outputs body).eraseDType.toAlgorithm? = (do
      return Kernel.mk inputs outputs (← erasedBody body)) := by
  cases h : ComputeStmt.listToAlgorithm? body <;>
    simp [ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, h, erasedBody,
      Kernel.eraseDType, ComputeKernel.fromKernelBody, bind, pure, Except.bind, Except.pure]

theorem erased_projection_of_ok (ck : ComputeKernel) (k : Kernel)
    (h : ck.toAlgorithm? = .ok k) :
    ck.eraseDType.toAlgorithm? = .ok k.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgorithm?_fromKernelBody]

theorem source_erased_body (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).eraseDType.toAlgorithm? = (do
      return Kernel.mk [] [y]
        (← erasedBody (source storage swapped y N seed offset lower upper B hB).surfaceBody)) := by
  cases storage <;> cases swapped
  all_goals exact erased_projection_mk [] [y] _

/-- All source operations preceding the stores are identical across storage
profiles. In particular the complete four-field random/affine prefix stays
opaque while proving the final storage erasure. -/
theorem prefix_same (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).surfaceBody.take 19 =
      (source .fp64 swapped y N seed offset lower upper B hB).surfaceBody.take 19 := by
  cases storage <;> cases swapped <;> rfl

theorem store_suffix_same (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    erasedBody ((source storage swapped y N seed offset lower upper B hB).surfaceBody.drop 19) =
      erasedBody ((source .fp64 swapped y N seed offset lower upper B hB).surfaceBody.drop 19) := by
  cases storage <;> cases swapped
  all_goals simp only [source, ComputeKernel.surfaceBody, List.drop_succ_cons, List.drop_zero,
    erasedBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, ComparableDType.eraseDType, Except.bind, bind, pure]
  all_goals simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

theorem source_erased_body_same (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    erasedBody (source storage swapped y N seed offset lower upper B hB).surfaceBody =
      erasedBody (source .fp64 swapped y N seed offset lower upper B hB).surfaceBody :=
  erasedBody_eq_of_parts _ _ 19
    (prefix_same storage swapped y N seed offset lower upper B hB)
    (store_suffix_same storage swapped y N seed offset lower upper B hB)

theorem complete_projection (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).eraseDType.toAlgorithm? =
      .ok (reference swapped y N seed offset lower upper B hB) := by
  rw [source_erased_body, source_erased_body_same]
  rw [← source_erased_body .fp64]
  exact erased_projection_of_ok _ _ (projection_ok .fp64 swapped y N seed offset lower upper B hB)

#axiomsClean projection_ok
#axiomsClean full_statement_count
#axiomsClean prefix_same
#axiomsClean store_suffix_same
#axiomsClean complete_projection

end VeriTile.Bench.TritonBenchG.UniformSampling.Erasure
