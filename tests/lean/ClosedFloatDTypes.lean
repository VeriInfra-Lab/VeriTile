import VeriTile.Triton.DSL.ClosedFloatDTypes
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ClosedFloatDTypes
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := []
  constexpr := ["dtype"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) : ComputeKernel :=
  let dtype : FloatDType := .fp16
  with_triton_call profile call_args [] in triton {
    value = tl.load(X).to(dtype)
    result = (value * 2.0).to(dtype)
    tl.store(Y, result)
  }

def explicitProfile : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := []
  constexpr := []
  readPorts := ["X"]
  writePorts := ["Y"]

/-- Every load, compound operand, explicit conversion and destination
conversion survives resolution of the original dtype parameter. -/
theorem complete_casts (X Y : RegionName) : copied X Y=
  with_triton_call explicitProfile call_args [] in triton {
    value = tl.load(X).to(tl.float16)
    result = (value * 2.0).to(tl.float16)
    tl.store(Y, result)
  } := rfl

#axiomsClean complete_casts
end
end VeriTile.Tests.ClosedFloatDTypes
