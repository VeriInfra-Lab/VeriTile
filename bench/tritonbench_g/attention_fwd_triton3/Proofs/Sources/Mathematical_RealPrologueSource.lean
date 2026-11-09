import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealPrologue

/-! Generic statement-list decomposition and precision-erasure identities. The actual pasted prologue is proved in Proofs/Prelude.lean and Window.lean. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealPrologueSource
open VeriTile.Triton

def beforeLoop : List Stmt → List Stmt
  | .forRangeInt _ _ _ _ _ :: _ => []
  | st :: rest => st :: beforeLoop rest
  | [] => []

def throughLoop : List Stmt → List Stmt
  | (.forRangeInt name lo hi stride body) :: _ => [.forRangeInt name lo hi stride body]
  | st :: rest => st :: throughLoop rest
  | [] => []

def pieces (body : List Stmt) : List Stmt :=
  RealHeaderSource.leadingAssignments body ++ (RealInitSource.branch? "m_i" body).toList ++
    RealInitSource.assignments "qk_scale" body ++ (RealInitSource.branch? "q" body).toList ++
    (RealWindowSource.window? body).toList

theorem before_loop_erase (body : List Stmt) :
    beforeLoop (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (beforeLoop body) := by
  induction body with
  | nil => simp [beforeLoop, Stmt.eraseDTypeList]
  | cons st rest ih => cases st <;> simp [beforeLoop, Stmt.eraseDTypeList, Stmt.eraseDType, ih]

theorem through_loop_erase (body : List Stmt) :
    throughLoop (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (throughLoop body) := by
  induction body with
  | nil => simp [throughLoop, Stmt.eraseDTypeList]
  | cons st rest ih => cases st <;> simp [throughLoop, Stmt.eraseDTypeList, Stmt.eraseDType, ih]

theorem option_list_erase (st : Option Stmt) :
    (st.map Stmt.eraseDType).toList = Stmt.eraseDTypeList st.toList := by
  cases st <;> simp [Stmt.eraseDTypeList]

theorem pieces_erase (body : List Stmt) :
    pieces (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (pieces body) := by
  simp only [pieces, RealHeaderSource.prefix_erase, RealInitSource.branch_erase,
    RealInitSource.assignments_erase, RealWindowSource.window_erase, option_list_erase,
    RealHeaderSource.erase_append]

theorem erased_pieces (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel)
    (hp : beforeLoop k.toAlgKernel.body = pieces k.toAlgKernel.body) :
    beforeLoop k.eraseDType.toAlgKernel.body = pieces k.eraseDType.toAlgKernel.body := by
  rw [RealInitSource.erased_body k h]
  exact ((before_loop_erase _).trans (congrArg Stmt.eraseDTypeList hp)).trans (pieces_erase _).symm

theorem erased_computation (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel)
    (hp : throughLoop k.toAlgKernel.body =
      beforeLoop k.toAlgKernel.body ++ (RealLoopSource.loop? "start_n" k.toAlgKernel.body).toList) :
    throughLoop k.eraseDType.toAlgKernel.body = beforeLoop k.eraseDType.toAlgKernel.body ++
      (RealLoopSource.loop? "start_n" k.eraseDType.toAlgKernel.body).toList := by
  rw [RealInitSource.erased_body k h, through_loop_erase, hp]
  simp only [before_loop_erase, RealLoopSource.loop_erase, option_list_erase, RealHeaderSource.erase_append]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealPrologueSource
