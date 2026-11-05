import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealInit

/-! Generic initialization-fragment erasure identities and syntax inspectors. These helpers do not define an alternative task kernel. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInitSource
open VeriTile.Triton

def firstAssignment? : List Stmt → Option RegName
  | .assign _ _ name _ :: _ => some name
  | _ => none

def branch? (name : RegName) : List Stmt → Option Stmt
  | [] => none
  | .ifThenElse c yes no :: rest =>
      if firstAssignment? yes = some name ∧ firstAssignment? no = some name
      then some (.ifThenElse c yes no) else branch? name rest
  | _ :: rest => branch? name rest

def assignments (name : RegName) : List Stmt → List Stmt
  | [] => []
  | (.assign dt shape dest value) :: rest =>
      if dest = name then .assign dt shape dest value :: assignments name rest else assignments name rest
  | _ :: rest => assignments name rest

theorem first_assignment_erase (code : List Stmt) :
    firstAssignment? (Stmt.eraseDTypeList code) = firstAssignment? code := by
  cases code with
  | nil => simp [Stmt.eraseDTypeList, firstAssignment?]
  | cons st rest => cases st <;> simp [Stmt.eraseDTypeList, Stmt.eraseDType, firstAssignment?]

theorem branch_erase (name : RegName) (code : List Stmt) :
    branch? name (Stmt.eraseDTypeList code) = (branch? name code).map Stmt.eraseDType := by
  induction code with
  | nil => simp [Stmt.eraseDTypeList, branch?]
  | cons st rest ih =>
    cases st <;> simp only [Stmt.eraseDTypeList, Stmt.eraseDType, branch?]
    all_goals try exact ih
    rw [first_assignment_erase, first_assignment_erase]
    split <;> simp_all [Stmt.eraseDType]

theorem assignments_erase (name : RegName) (code : List Stmt) :
    assignments name (Stmt.eraseDTypeList code) = Stmt.eraseDTypeList (assignments name code) := by
  induction code with
  | nil => simp [Stmt.eraseDTypeList, assignments]
  | cons st rest ih =>
    cases st <;> simp only [Stmt.eraseDTypeList, Stmt.eraseDType, assignments]
    all_goals try exact ih
    split <;> simp_all [Stmt.eraseDTypeList, Stmt.eraseDType]

theorem erased_body (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel.body = Stmt.eraseDTypeList k.toAlgKernel.body := by
  rw [ComputeKernel.eraseDType, h]
  simp only [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody, Kernel.eraseDType]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInitSource
