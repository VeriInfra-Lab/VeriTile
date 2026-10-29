import bench.repairs.tritonbench_g.kldiv_ops.Proofs.HostContract
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Contract
namespace VeriTile.Bench.Repairs.KldivOps.Contiguous

private theorem forward_iff (a b : Tensor) (eps : Bool) :
    ForwardContract a b eps ↔ PairContract a b ∧ eps = Bool.true := by
  constructor
  · intro h;exact ⟨h.toPairContract,h.epsilon⟩
  · rintro ⟨a,b⟩;exact ⟨a,b⟩

theorem python_forward_checks_iff (a b : Argument) (flag : Option Bool) (mode : String) (eps : Bool) :
    checkForward a b flag mode eps = .ok () ↔ AdmittedForward a b flag mode eps := by
  have bind_iff (first next : Except String Unit) :
      first.bind (fun _ => next) = .ok () ↔ first = .ok () ∧ next = .ok () := by
    cases first with
    | error e => simp [Except.bind]
    | ok v => cases v;simp [Except.bind]
  unfold checkForward
  change (KldivTriton.Contiguous.checkForward a b flag mode).bind (fun _ => _) = .ok () ↔ _
  rw [bind_iff,KldivTriton.Contiguous.python_forward_checks_iff]
  cases a <;> cases b <;> cases eps <;>
    simp [AdmittedForward,KldivTriton.Contiguous.AdmittedForward,forward_iff, and_assoc]

abbrev python_backward_checks_iff := KldivTriton.Contiguous.python_backward_checks_iff
#axiomsClean python_forward_checks_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivOps.Contiguous
