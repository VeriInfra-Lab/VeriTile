import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Memory
import bench.tritonbench_g.relu_strided_buffer.Kernels

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer HostPlan
noncomputable section

/-- Admission precedes the unchanged wrapper body. The ordinary CUDA guard
can still fail; no launch callback supplies output data. -/
def runChecked [Inhabited α] (M : Algebra α) (x y : ViewMetadata) (guardSucceeds : Bool)
    (s : State α) : Except InputError (StructuralHost.Result α) := do
  checkInputs x y
  pure (Whole.runHost M x.buffer y.buffer guardSucceeds s)

theorem checked_eq_source [Inhabited α] (M : Algebra α) (x y : ViewMetadata)
    (admitted : checkInputs x y = .ok ()) (guard : Bool) (s : State α) :
    runChecked M x y guard s = .ok (Whole.runHost M x.buffer y.buffer guard s) := by
  simp [runChecked,admitted]

/-- Optional historical host/source bridge. Public acceptance uses kernels only. -/
theorem launch_uses_source [Inhabited α] (M : Algebra α) (a : Arguments) (s : State α) :
    StructuralHost.executeArguments M a s =
      StructuralControl.runPrograms M (Whole.kernel a).surfaceBody a.C (List.range a.C) s := rfl

#axiomsClean launch_uses_source
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
