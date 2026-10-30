import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.InputContract

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.Contract
open RmsMatmulRbe.Contract (dim Error)
abbrev Inputs := RmsRbeMatmul.Contract.Inputs

/-- The unchanged source uses batch on axis x and flattened M/N tiles on axis y. -/
def grid (i : Inputs) : Nat × Nat :=
  (dim i.x 0, ((dim i.x 1 + 15) / 16) * ((dim i.weight 0 + 63) / 64))

/-- Empty wrappers skip launch. Only the two actual nonempty launch axes are bounded. -/
def LaunchAdmitted (i : Inputs) : Prop :=
  0 < dim i.x 0 → 0 < dim i.x 1 → (grid i).1 ≤ 2^31-1 ∧ (grid i).2 ≤ 65535
instance (i : Inputs) : Decidable (LaunchAdmitted i) :=
  inferInstanceAs (Decidable (_ → _ → _ ∧ _))

def InputContract (i : Inputs) : Prop :=
  RmsRbeMatmul.Contract.InputContract i ∧ LaunchAdmitted i

def launchGuard (i : Inputs) : Bool :=
  !(decide (0 < dim i.x 0) && decide (0 < dim i.x 1) &&
    (decide (2^31-1 < (grid i).1) || decide (65535 < (grid i).2)))

noncomputable def checkInputs (i : Inputs) : Except Error Unit :=
  match RmsRbeMatmul.Contract.checkInputs i with
  | .error e => .error e
  | .ok () => if launchGuard i then .ok () else .error .shape

theorem launchGuard_iff (i : Inputs) : launchGuard i = true ↔ LaunchAdmitted i := by
  simp [launchGuard, LaunchAdmitted]
  omega

theorem checkInputs_iff (i : Inputs) : checkInputs i = .ok () ↔ InputContract i := by
  unfold checkInputs InputContract
  cases result : RmsRbeMatmul.Contract.checkInputs i with
  | error e =>
    have rejected : ¬ RmsRbeMatmul.Contract.InputContract i := by
      rw [← RmsRbeMatmul.Contract.checkInputs_iff, result]
      simp
    simp [rejected]
  | ok unit =>
    cases unit
    have admitted := (RmsRbeMatmul.Contract.checkInputs_iff i).mp result
    simp [admitted, launchGuard_iff]

theorem nonempty_axes (i : Inputs) (h : InputContract i)
    (batch : 0 < dim i.x 0) (rows : 0 < dim i.x 1) :
    (grid i).1 ≤ 2^31-1 ∧ (grid i).2 ≤ 65535 := h.2 batch rows

theorem empty_batch (i : Inputs) (empty : dim i.x 0 = 0) : LaunchAdmitted i := by
  simp [LaunchAdmitted, empty]

theorem empty_rows (i : Inputs) (empty : dim i.x 1 = 0) : LaunchAdmitted i := by
  simp [LaunchAdmitted, empty]

#axiomsClean launchGuard_iff
#axiomsClean checkInputs_iff
#axiomsClean nonempty_axes
#specNonCircular InputContract avoiding [checkInputs]
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.Contract
