import bench.repairs.tritonbench_g.swiglu_triton.Proofs.InputGuards
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.DirectSource

namespace VeriTile.Bench.Repairs.SwigluTriton.InputBinding
open VeriTile.Triton FP.Structural Memory DirectHost DirectPhysical
noncomputable section

/-- The checked payloads and the unchanged calculate_settings block limit. -/
def InputContract (p : Prepared backward) : Prop :=
  InputAdmitted p ∧ blockSize p.cols ≤ 65536

def checkInputs (p : Prepared backward) : Bool :=
  inputCheck p && decide (blockSize p.cols ≤ 65536)

theorem checkInputs_iff (p : Prepared backward) :
    checkInputs p = Bool.true ↔ InputContract p := by
  simp only [checkInputs, InputContract, Bool.and_eq_true, decide_eq_true_eq, inputCheck_iff]

/-- One actual memory reader for the checked pointer views. Existing Tensor
laws include the storage offset in dataPtr; input values are not independent
arrays. The snapshot changes only entry observations, not host execution. -/
def initial {α : Type} [Inhabited α] (p : Prepared backward)
    (read : Key → α) (scratch : State α) : State α :=
  { scratch with mem := fun r o =>
      if r = "A" then .mk .real (read ⟨(p.tensor .a).metadata.device.index,
        (p.tensor .a).metadata.dataPtr.toNat + 4*o⟩)
      else if r = "B" then .mk .real (read ⟨(p.tensor .b).metadata.device.index,
        (p.tensor .b).metadata.dataPtr.toNat + 4*o⟩)
      else if r = DirectSource.region backward .third then .mk .real (read ⟨(p.tensor .third).metadata.device.index,
          (p.tensor .third).metadata.dataPtr.toNat + 4*o⟩)
      else scratch.mem r o }

theorem logical_read {α : Type} [Inhabited α] (p : Prepared backward)
    (h : InputAdmitted p) (read : Key → α) (scratch : State α)
    (role : Role) (i : Nat) (hi : i < p.rows*p.cols) :
    (initial p read scratch).mem (DirectSource.region backward role) i =
      Cell.mk .real (read ((p.tensor role).key i)) := by
  have admitted := input_admitted p h
  rw [(p.tensor role).contiguousKeys (strided p admitted role) (fp32 p admitted role)
    (contiguous p admitted role) i (by rwa [admitted.counts role])]
  cases backward <;> cases role <;> simp [initial, DirectSource.region]

#axiomsClean logical_read
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.InputBinding
