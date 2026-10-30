import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeSource

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
open VeriTile.Triton FP.Structural WholeSource
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open HostPlan
noncomputable section
local instance : Inhabited Float32Bits := ⟨⟨0⟩⟩

/-- The selected source scalar kind is an ordinary Python float. Its actual
IEEE binary64 payload is retained, including NaN, infinity and signed zero.
Python integer and tensor scalar specializations are separate source profiles. -/
structure PythonFloat where
  payload : Float64Bits

/-- The sole host numeric primitive. It represents the pinned launcher's
PyFloat_AsDouble followed by the C float conversion and raw-word extraction.
It supplies one scalar argument, never any tensor result/control/address.
This is an ordinary launch-ABI model boundary, not a task-specific numerical
experiment. It is not real identity: finite Python values can round, underflow
or overflow here. Kernel-only proofs start from the supplied FP32 scalar word. -/
structure ScalarABI where
  extractFP32 : Float64Bits → Float32Bits

/-- Source `.const` is the existing DSL encoding of the symbolic val0 argument.
There is exactly one floating scalar occurrence in this selected source. This
fixed key is interpreted by fp32Backend as the extracted host argument; it is
not the floating numeric value zero. Real uses the independent real base key. -/
def sourceSlot : ℝ := 0

/-- `baseWord` is the runtime FP32 scalar argument after host specialization.
The sole floating literal in this kernel denotes that argument. All three
source casts have FP32 input/output representation and preserve every word,
including both zeros and NaN payloads. Power remains the exact opaque primitive
slot; no law equates it to real power or to exp/log arithmetic. -/
def fp32Backend (rest : Algebra Float32Bits) (baseWord : Float32Bits) : Algebra Float32Bits :=
  { rest with literal := fun _ _ _ => baseWord
              cast := fun _ _ _ word => word }

/-- Exact original preparation order. No extraction occurs on shape, empty,
metadata or CUDA-guard failure; extraction is the scalar launch-argument step.
The extracted word binds the only val0 occurrence before complete grid execution. -/
def runPythonHost (rest : Algebra Float32Bits) (abi : ScalarABI) (host : PythonFloat)
    (candidate : Bool) (input output : Buffer) (guardSucceeds : Bool)
    (s : State Float32Bits) : Result Float32Bits :=
  match prepare input output guardSucceeds with
  | .error failure => .hostFailure failure s
  | .ok args =>
    let runtimeBase := abi.extractFP32 host.payload
    match executeArguments (fp32Backend rest runtimeBase) sourceSlot candidate args s with
    | none => .executionFailure
    | some final => .returned output final

theorem python_run_eq (rest : Algebra Float32Bits) (abi : ScalarABI) (host : PythonFloat)
    (candidate : Bool) (input output : Buffer) (guardSucceeds : Bool) (s : State Float32Bits) :
    runPythonHost rest abi host candidate input output guardSucceeds s =
      runHost (fp32Backend rest (abi.extractFP32 host.payload)) sourceSlot candidate
        input output guardSucceeds s := by
  unfold runPythonHost runHost
  cases hprep : prepare input output guardSucceeds with
  | error failure => rfl
  | ok args =>
    simp only
    cases hexec : executeArguments (fp32Backend rest (abi.extractFP32 host.payload)) sourceSlot candidate args s <;> simp only [hexec]

theorem python_host_failure (rest : Algebra Float32Bits) (abi : ScalarABI) (host : PythonFloat)
    (candidate : Bool) (input output : Buffer) (guardSucceeds : Bool) (s : State Float32Bits)
    (failure : Failure) (failed : prepare input output guardSucceeds = .error failure) :
    runPythonHost rest abi host candidate input output guardSucceeds s = .hostFailure failure s := by
  simp only [runPythonHost, failed]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
