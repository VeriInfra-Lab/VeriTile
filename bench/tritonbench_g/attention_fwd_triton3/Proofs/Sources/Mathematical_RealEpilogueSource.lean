import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealEpilogue

/-! Generic statement-list splitting and precision-erasure identities for mathematical store fragments. Actual padded source stores are proved in Proofs/Epilogue.lean. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealEpilogueSource
open VeriTile.Triton RealPrologueSource

def afterLoop : List Stmt → List Stmt
  | .forRangeInt _ _ _ _ _ :: rest => rest
  | _ :: rest => afterLoop rest
  | [] => []

theorem after_loop_erase (body : List Stmt) :
    afterLoop (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (afterLoop body) := by
  induction body with
  | nil => simp [afterLoop, Stmt.eraseDTypeList]
  | cons st rest ih => cases st <;> simp [afterLoop, Stmt.eraseDTypeList, Stmt.eraseDType, ih]

theorem through_after_loop (body : List Stmt) : throughLoop body ++ afterLoop body = body := by
  induction body with
  | nil => rfl
  | cons st rest ih => cases st <;> simp [throughLoop, afterLoop, ih]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealEpilogueSource
