import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Real
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Checked.Entry

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Checked
noncomputable section

/-- Independent dense target for the actual checked source call. Successful
returns include the actual descriptor and device state and a successful call
to save_for_backward with converted operands in original source order. -/
specification correct (ts : Fin 4 → Observed) (flag : Bool) (binding : Binding r ts)
    (allocate : DeviceState σ → Result σ Unit) (abi : LocalEventABI ℝ σ realArithmetic)
    (rt : Tail) (h : Allocation r caller (before : Memory ℝ))
    (allocLaw : AllocationLaw abi h allocate) (tailLaw : TailContract rt h)
    (failures : Failures) (trials : List (Fin 98)) (c : Fin 98)
    (state after : DeviceState σ) (out : Tensor) (entry : abi.observe state.payload = before)
    (success : call ts flag allocate abi rt h failures trials c state = .ok out after) :
    Domain r ts flag ∧ out = h.returned ∧
    (∀ m n, m < r.batch.rows → n < r.n →
      abi.observe after.payload (h.storage 0,outputOffset r.shape m n) = denseOutput r before m n) ∧
    (∀ m n, m < r.batch.rows → n < r.n → r.hasSaved = Bool.true →
      abi.observe after.payload (r.storage 3,outputOffset r.shape m n) = densePre r before m n) ∧
    (∀ p, p.1 ∈ caller → ¬(r.hasSaved = Bool.true ∧ p.1 = r.storage 3 ∧ p.2 < r.batch.rows*r.n) →
      abi.observe after.payload p = before p) ∧
    after.currentDevice = state.currentDevice ∧ rt.save h.context = .ok () := by
  obtain ⟨admit,returned,memory,device,saved⟩ :=
    call_success ts flag binding allocate abi rt h allocLaw tailLaw failures trials c state after out entry success
  have reference := whole_real h c (domain_valid r ts binding admit) state.currentDevice
  exact ⟨admit,returned,by simpa only [memory] using reference.1,
    by simpa only [memory] using reference.2.1,
    by simpa only [memory] using reference.2.2.1,device,saved⟩

#axiomsClean correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Checked
