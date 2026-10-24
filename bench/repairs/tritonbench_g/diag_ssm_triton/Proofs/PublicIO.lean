import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.TensorLaws

namespace VeriTile.Bench.Repairs.DiagSsmTriton.PublicIO
open Kernels Tensors Host LibraryContracts TensorLaws Execution
set_option autoImplicit false

/-- All matching FP32 or complex64 tensors of the documented logical shapes.
This internal IO interface keeps the source contiguity condition; the public
checked entries additionally require absent lazy flags. Nonzero storage offsets
and read-only aliases are allowed.
There is no finite-value or newly imposed native-safe shape predicate. -/
structure Forward (route : Route) (d : Dimensions) (s x transition : Tensor) (before : State) : Prop where
  initial : Layout route [d.batch, d.dim] s
  input : Layout route [d.length, d.batch, d.dim] x
  parameter : Layout route [d.dim] transition
  initialLive : Resident before s
  inputLive : Resident before x
  parameterLive : Resident before transition

/-- Backward receives the actual saved forward descriptors and an arbitrary
layout of matching cotangents. Version checks are performed by restore; this
predicate does not assume that the version check or any later API succeeds.
All saved tensors remain live under the context's ordinary ownership. -/
structure Backward (route : Route) (d : Dimensions) (saved : Saved) (gy : Tensor) (before : State) : Prop where
  present : before.saved = some saved
  initial : Layout route [d.batch, d.dim] saved.initial
  output : Layout route [d.length, d.batch, d.dim] saved.output
  parameter : Layout route [d.dim] saved.transition
  gradientRoute : gy.route = route
  gradientShape : gy.shape = [d.length, d.batch, d.dim]
  gradientLive : Resident before gy
  outputLive : Resident before saved.output
  initialLive : Resident before saved.initial
  parameterLive : Resident before saved.transition

end VeriTile.Bench.Repairs.DiagSsmTriton.PublicIO
