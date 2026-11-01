import VeriTile.Triton.DSL.PythonInlineSuites
namespace VeriTile.Tests.BinaryPureJitReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("Out",.i64)]
  scalars := [("N",.i64)]
  constexpr := ["B"]
  writePorts := ["Out"]

/-- Independently elaborated integer reference; in particular no return
conversion or floating computation is present. -/
def reference (Out : RegionName) (N : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    __veritile_binary_jit_0 = N
    __veritile_binary_jit_1 = B
    __veritile_binary_jit_2 = tl.cdiv(__veritile_binary_jit_0, __veritile_binary_jit_1) * __veritile_binary_jit_1 - __veritile_binary_jit_1
    previous = __veritile_binary_jit_2
    tl.store(Out, previous)
  }
/-- Source-private name collision plus two calls: fresh captures must be
independent of each other and of the caller's reserved-looking name. -/
def repeatedReference (Out : RegionName) (N : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    __veritile_binary_jit_0 = N
    __veritile_binary_jit_1 = N
    __veritile_binary_jit_2 = B
    __veritile_binary_jit_3 = tl.cdiv(__veritile_binary_jit_1, __veritile_binary_jit_2) * __veritile_binary_jit_2 - __veritile_binary_jit_2
    first = __veritile_binary_jit_3
    __veritile_binary_jit_4 = N
    __veritile_binary_jit_5 = B
    __veritile_binary_jit_6 = tl.cdiv(__veritile_binary_jit_4, __veritile_binary_jit_5) * __veritile_binary_jit_5 - __veritile_binary_jit_5
    second = __veritile_binary_jit_6
    tl.store(Out, first + second + __veritile_binary_jit_0)
  }
end VeriTile.Tests.BinaryPureJitReference
