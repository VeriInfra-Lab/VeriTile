import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealPointers

/-! Generic header-fragment erasure identities and syntax inspectors. Actual caller/helper capture and execution are checked in the current Prelude and Window proofs. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealHeaderSource
open VeriTile.Triton

def leadingAssignments : List Stmt → List Stmt
  | .assign dt shape name value :: rest => .assign dt shape name value :: leadingAssignments rest
  | _ => []

theorem prefix_erase (body : List Stmt) :
    leadingAssignments (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (leadingAssignments body) := by
  induction body with
  | nil => simp [leadingAssignments, Stmt.eraseDTypeList]
  | cons st rest ih => cases st <;> simp [leadingAssignments, Stmt.eraseDTypeList, Stmt.eraseDType, ih]

theorem erased_prefix (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    leadingAssignments k.eraseDType.toAlgKernel.body = Stmt.eraseDTypeList (leadingAssignments k.toAlgKernel.body) := by
  rw [RealInitSource.erased_body k h]
  exact prefix_erase _

theorem erase_append (a b : List Stmt) :
    Stmt.eraseDTypeList (a ++ b) = Stmt.eraseDTypeList a ++ Stmt.eraseDTypeList b := by
  induction a with
  | nil => simp [Stmt.eraseDTypeList]
  | cons st rest ih => simp [Stmt.eraseDTypeList, ih]

theorem coordinates_erase (heads kvHeads : Nat) (p : RealHeader.Strides) :
    Stmt.eraseDTypeList (RealHeader.code heads kvHeads p) = RealHeader.code heads kvHeads p := by
  simp only [RealHeader.code, RealHeader.coordinateCode, RealHeader.offsetCode, RealHeader.offsetExpr,
    List.cons_append, List.nil_append, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, IntegralDType.eraseDType, VeriTile.Triton.eraseDType]

theorem dynamic_rows_erase (region : RegionName) (name : RegName)
    (BM BD context strideRow strideDim : Nat) :
    (Op.makeBlockPtrDynOffsets region (.ref .nat [] name) [context, BD] [BM, BD]
      [strideRow, strideDim] [RealPointers.rowOffsetExpr BM, .constNat 0]).eraseDType =
    .makeBlockPtrDynOffsets region (.ref .nat [] name) [context, BD] [BM, BD]
      [strideRow, strideDim] [RealPointers.rowOffsetExpr BM, .constNat 0] := by
  rw [Op.eraseDType.eq_def]
  change Op.makeBlockPtrDynOffsets region (Op.ref .nat [] name).eraseDType [context, BD] [BM, BD]
    [strideRow, strideDim] [(RealPointers.rowOffsetExpr BM).eraseDType, (Op.constNat 0).eraseDType] = _
  simp only [RealPointers.rowOffsetExpr, Op.eraseDType.eq_def, NumericDType.eraseDType]
  rfl

theorem pointers_erase (BM BN BD : Nat) (p : RealPointers.Layout) :
    Stmt.eraseDTypeList (RealPointers.code BM BN BD p) = RealPointers.code BM BN BD p := by
  simp only [RealPointers.code, Stmt.eraseDTypeList, Stmt.eraseDType, RealPointers.queryExpr, RealPointers.outputExpr]
  rw [dynamic_rows_erase, dynamic_rows_erase]
  simp only [RealPointers.valueExpr, RealPointers.keyExpr, RealPointers.rowPointerExpr, RealPointers.rowOffsetExpr,
    Op.eraseDType.eq_def, NumericDType.eraseDType, VeriTile.Triton.eraseDType]
  rfl

theorem code_erase (BM BN BD heads kvHeads : Nat) (p : RealHeader.Strides) (m : RealPointers.Layout) :
    Stmt.eraseDTypeList (RealHeader.code heads kvHeads p ++ RealPointers.code BM BN BD m) =
      RealHeader.code heads kvHeads p ++ RealPointers.code BM BN BD m := by
  rw [erase_append, coordinates_erase, pointers_erase]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealHeaderSource
