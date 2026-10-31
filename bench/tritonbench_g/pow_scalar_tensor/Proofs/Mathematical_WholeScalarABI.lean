import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeSource

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

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
