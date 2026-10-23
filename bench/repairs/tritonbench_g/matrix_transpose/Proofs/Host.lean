import bench.repairs.tritonbench_g.matrix_transpose.Proofs.FP
import bench.tritonbench_g.matrix_transpose.FullScope

/-! The four guards precede the unchanged original wrapper. Existing complete
source-order outcomes retain ordinary allocation/launch failures; their success
case is established by actual typed source execution. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixTranspose
noncomputable section

def runArguments (m d : Argument) (initial : State FP.Word) (samples : Nat → FP.Word)
    (failures : FullScope.PrimitiveFailures) : Except InputError FullScope.Outcome := do
  checkInputs m d
  match m, d with
  | .integer a, .integer b =>
    return FullScope.wrapper (FP.parameters a.toNat b.toNat) initial samples failures
  | _, _ => .error .integers

/-- Independent source-order postcondition: an exact guard rejection, or the
original allocation/launch failure, or the exact FP16 transpose and full frame.
The inherited postcondition contains no source execution or success premise. -/
def Post (m d : Argument) (initial : State FP.Word) (samples : Nat → FP.Word)
    (failures : FullScope.PrimitiveFailures) (result : Except InputError FullScope.Outcome) : Prop :=
  match m, d with
  | .integer a, .integer b =>
    match checkIntegers a b with
    | .error reason => result = .error reason
    | .ok _ => ∃ outcome,
      result = .ok outcome ∧
      FullScope.WrapperPost (FP.parameters a.toNat b.toNat) initial samples failures outcome
  | _, _ => result = .error .integers

theorem all_outcomes (m d : Argument) (initial : State FP.Word) (samples : Nat → FP.Word)
    (failures : FullScope.PrimitiveFailures) :
    Post m d initial samples failures (runArguments m d initial samples failures) := by
  cases m with
  | other => cases d <;> rfl
  | integer a =>
    cases d with
    | other => rfl
    | integer b =>
      simp only [Post, runArguments, checkInputs]
      cases checked : checkIntegers a b with
      | error reason => rfl
      | ok value =>
        cases value
        refine ⟨FullScope.wrapper (FP.parameters a.toNat b.toNat) initial samples failures, rfl, ?_⟩
        exact FullScope.matrix_transpose_full_source_outcomes _ initial samples failures
          (by change ("matrix" : RegionName) ≠ "out"; decide)

/-- The guards retain the entire original frontend-success shape domain. -/
theorem contract_iff_original_frontend (m d : Nat) :
    InputContract m d ↔ FullScope.admission m d = none := by
  rw [FullScope.admission_none]
  exact contract_iff_sourceShape m d

/-- The wrapper's admitted route reaches no frontend-shape rejection. -/
theorem accepted_frontend (a b : Int) (h : checkIntegers a b = .ok ()) :
    FullScope.admission a.toNat b.toNat = none := by
  apply (FullScope.admission_none a.toNat b.toNat).2
  exact (contract_iff_sourceShape a.toNat b.toNat).1 ((checkIntegers_iff a b).1 h).2.2

/-- The public metadata is exactly the original returned allocation's metadata
on admitted dimensions, including its region identity and both strides. -/
theorem returned_metadata (m d : Nat) (h : InputContract m d) :
    (FullScope.returnedTensor (FP.parameters m d)).region = (returnedTensor m d).region ∧
    (FullScope.returnedTensor (FP.parameters m d)).shape = (returnedTensor m d).shape ∧
    (FullScope.returnedTensor (FP.parameters m d)).strides = (returnedTensor m d).strides ∧
    (FullScope.returnedTensor (FP.parameters m d)).storage = (returnedTensor m d).storage ∧
    (FullScope.returnedTensor (FP.parameters m d)).cuda = (returnedTensor m d).cuda := by
  simp [FullScope.returnedTensor, FP.parameters, returnedTensor,
    VeriTile.Bench.TritonBenchG.MatrixTranspose.Host.outputShape,
    VeriTile.Bench.TritonBenchG.MatrixTranspose.Host.outputStrides,
    Nat.max_eq_left h.positive.1]

#axiomsClean all_outcomes
#axiomsClean contract_iff_original_frontend
#axiomsClean accepted_frontend
#axiomsClean returned_metadata
#specNonCircular Post avoiding [runArguments, FullScope.wrapper, FP.runWrapper, FPBits.execution]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixTranspose.Host
