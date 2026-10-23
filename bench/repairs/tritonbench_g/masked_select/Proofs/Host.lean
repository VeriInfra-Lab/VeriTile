import bench.repairs.tritonbench_g.masked_select.Proofs.Contract
import bench.tritonbench_g.masked_select.Proofs.HostComposition

namespace VeriTile.Bench.Repairs.MaskedSelect
open VeriTile.Triton FP.Structural
open VeriTile.Torch
open VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy
open VeriTile.Bench.TritonBenchG.MaskedSelect.Whole
noncomputable section

/-- Successful-primitive wrapper interpretation. preparedPointer is the actual
pointer returned by resolve_neg().contiguous(), bound to NativeBuffers in the
FP entry. The guard occurs after that producer and before output allocation/launch.
The producer's logical effect is unchanged by resolving a negative view. -/
def runChecked {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (x m : TensorMetadata) (input : Tensor α) (mask : Tensor Bool)
    (preparedPointer : Nat) (trials : List LaunchPlan) (selected : LaunchPlan)
    (s : State α) : Except InputError (Except HostFailure (Returned × State α)) := do
  let shape ← checkTensorInputs x m
  let q := {p with size := shape.prod}
  let produced := prepare q (broadcastValues input shape) (broadcastValues mask shape) s
  checkPrepared preparedPointer
  return match runLaunches rest q trials produced with
    | none => .error .kernelExecution
    | some trialState => match runGrid rest (parametersFor q selected) selected.jobs trialState with
      | none => .error .kernelExecution
      | some finalState => .ok (returned q input.device (prefixLast q produced), finalState)

theorem checked_eq_original {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (x m : TensorMetadata) (input : Tensor α) (mask : Tensor Bool) (shape : List Nat)
    (h : InputContract x m shape) (preparedPointer : Nat) (hp : PreparedContract preparedPointer)
    (inputShape : input.shape = x.shape) (maskShape : mask.shape = m.shape)
    (trials : List LaunchPlan) (selected : LaunchPlan) (s : State α) :
    runChecked rest p x m input mask preparedPointer trials selected s =
      .ok (runHost rest p input mask trials selected s) := by
  have hb : broadcastShape input.shape mask.shape = some shape := by
    rw [inputShape,maskShape]; exact h.broadcast
  simp [runChecked,runHost,(checkTensorInputs_iff x m shape).mpr h,
    (checkPrepared_iff preparedPointer).mpr hp,hb,Nat.ne_of_gt h.extent.1,bind,Except.bind,pure,Except.pure]
  rfl

#axiomsClean checked_eq_original
end
end VeriTile.Bench.Repairs.MaskedSelect
