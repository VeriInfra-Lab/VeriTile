import Mathlib.Tactic
import bench.tritonbench_g.kldiv_triton.Proofs.HostContract
namespace VeriTile.Bench.Repairs.KldivTriton.Contiguous
open VeriTile.Torch
set_option maxHeartbeats 4000000

private theorem tensor_iff (t : Tensor) : TensorContract t ↔
    t.metadata.layout = .strided ∧ t.metadata.dtype = .floating .fp32 ∧
    t.metadata.device.kind = "cuda" ∧ t.metadata.isContiguous = Bool.true ∧
    t.metadata.isNeg = Bool.false ∧ (t.metadata.numel ≠ 0 → t.metadata.dataPtr.toNat % 4 = 0) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.contiguous,h.noNeg,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f⟩; exact ⟨a,b,c,d,e,f⟩

private theorem pair_iff (a b : Tensor) : PairContract a b ↔
    TensorContract a ∧ TensorContract b ∧ b.metadata.device = a.metadata.device ∧
    a.metadata.shape.length = 2 ∧ b.metadata.shape.length = 2 ∧ b.metadata.shape = a.metadata.shape := by
  constructor
  · intro h; exact ⟨h.left,h.right,h.device,h.leftRank,h.rightRank,h.shape⟩
  · rintro ⟨a,b,c,d,e,f⟩; exact ⟨a,b,c,d,e,f⟩

private theorem backward_iff (a b g : Tensor) : BackwardContract a b g ↔
    PairContract a b ∧ TensorContract g ∧ g.metadata.device = a.metadata.device ∧
    (g.metadata.shape = [] ∨ g.metadata.shape = a.metadata.shape) ∧
    (a.metadata.numel ≠ 0 → StorageSeparate a b ∧ StorageSeparate a g) := by
  constructor
  · intro h; exact ⟨h.toPairContract,h.gradient,h.gradientDevice,h.gradientShape,h.separate⟩
  · rintro ⟨a,b,c,d,e⟩; exact ⟨a,b,c,d,e⟩

private theorem list_iff (ts : List Tensor) : checkTensorList ts = .ok () ↔
    (∀ t ∈ ts, TensorContract t) ∧ (match ts with | [] => True | a::rest => ∀ b ∈ rest, b.metadata.device = a.metadata.device) := by
  classical
  simp only [checkTensorList, tensor_iff]
  split_ifs <;> simp_all
  all_goals aesop

private theorem pair_checks (a b : Tensor) : checkPair a b = .ok () ↔
    a.metadata.shape.length = 2 ∧ b.metadata.shape.length = 2 ∧ b.metadata.shape = a.metadata.shape := by
  classical
  simp only [checkPair]
  split_ifs <;> simp_all

private theorem bind_iff (a b : Except String Unit) :
    a.bind (fun _ => b) = .ok () ↔ a = .ok () ∧ b = .ok () := by
  cases a with
  | error e => simp [Except.bind]
  | ok u => cases u; simp [Except.bind]

/-- Exactly the checked Python forward domain, including classification. -/
theorem python_forward_checks_iff (a b : Argument) (flag : Option Bool) (mode : String) :
    checkForward a b flag mode = .ok () ↔ AdmittedForward a b flag mode := by
  classical
  cases a <;> cases b <;> simp only [checkForward, AdmittedForward]
  all_goals try simp
  next a b =>
    change (checkTensorList [a,b]).bind (fun _ => (checkPair a b).bind (fun _ => _)) = .ok () ↔ _
    rw [bind_iff,bind_iff,list_iff,pair_checks]
    cases flag <;> simp [ForwardContract,pair_iff,Option.isNone,Option.isSome]
    all_goals try split_ifs <;> simp_all
    all_goals tauto

/-- Exactly the checked Python backward domain; empty destinations need no alias check. -/
theorem python_backward_checks_iff (a b g : Argument) (flag : Option Bool) :
    checkBackward a b g flag = .ok () ↔ AdmittedBackward a b g flag := by
  classical
  cases a <;> cases b <;> cases g <;> simp only [checkBackward, AdmittedBackward]
  all_goals try simp
  next a b g =>
    change (checkTensorList [a,b,g]).bind (fun _ => (checkPair a b).bind (fun _ => _)) = .ok () ↔ _
    rw [bind_iff,bind_iff,list_iff,pair_checks]
    cases flag <;> simp_all [backward_iff,pair_iff,Option.isNone,Option.isSome]
    all_goals try split_ifs <;> simp_all
    all_goals tauto

#axiomsClean python_forward_checks_iff
#axiomsClean python_backward_checks_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivTriton.Contiguous
