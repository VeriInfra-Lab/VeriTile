import VeriTile.Triton.DSL.PythonInlineSuites
namespace VeriTile.Tests.PythonShapeTupleReference
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("Out",.fp32)]
  constexpr := ["B"]
  writePorts := ["Out"]

def reference (Out : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    values = tl.full([B], value=0, dtype=Out.dtype.element_ty)
    tl.store(Out + tl.arange(0, B), values)
  }
end
end VeriTile.Tests.PythonShapeTupleReference
