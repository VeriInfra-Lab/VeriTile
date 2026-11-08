import VeriTile.Triton.DSL.PythonInlineSuites
namespace VeriTile.Tests.FloatDTypeConstantReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.i64)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

/-- Independent typed reference compiled before the dtype-constant adapter. -/
def reference (X Y : RegionName) (N : Int) (B : Nat) : FloatDType → ComputeKernel
  | .real =>
    with_triton_call signature call_args [N := N] in triton {
      zero = tl.full([], tl.cast(0, tl.int64), dtype=tl.int64)
      group = tl.full([], tl.cast(B, tl.int64), dtype=tl.int64)
      one = tl.full([], tl.cast(1, tl.int64), dtype=tl.int64)
      last = N + group
      for row in range(tl.cast(N, tl.int64), tl.cast(last, tl.int64), tl.cast(1, tl.int64)):
        value = tl.load(X)
        tl.store(Y, value.to(tl.float64))
    }
  | .fp32 =>
    with_triton_call signature call_args [N := N] in triton {
      zero = tl.full([], tl.cast(0, tl.int64), dtype=tl.int64)
      group = tl.full([], tl.cast(B, tl.int64), dtype=tl.int64)
      one = tl.full([], tl.cast(1, tl.int64), dtype=tl.int64)
      last = N + group
      for row in range(tl.cast(N, tl.int64), tl.cast(last, tl.int64), tl.cast(1, tl.int64)):
        value = tl.load(X)
        tl.store(Y, value.to(tl.float32))
    }
  | .fp16 =>
    with_triton_call signature call_args [N := N] in triton {
      zero = tl.full([], tl.cast(0, tl.int64), dtype=tl.int64)
      group = tl.full([], tl.cast(B, tl.int64), dtype=tl.int64)
      one = tl.full([], tl.cast(1, tl.int64), dtype=tl.int64)
      last = N + group
      for row in range(tl.cast(N, tl.int64), tl.cast(last, tl.int64), tl.cast(1, tl.int64)):
        value = tl.load(X)
        tl.store(Y, value.to(tl.float16))
    }
  | .bf16 =>
    with_triton_call signature call_args [N := N] in triton {
      zero = tl.full([], tl.cast(0, tl.int64), dtype=tl.int64)
      group = tl.full([], tl.cast(B, tl.int64), dtype=tl.int64)
      one = tl.full([], tl.cast(1, tl.int64), dtype=tl.int64)
      last = N + group
      for row in range(tl.cast(N, tl.int64), tl.cast(last, tl.int64), tl.cast(1, tl.int64)):
        value = tl.load(X)
        tl.store(Y, value.to(tl.bfloat16))
    }
  | .f8e4 =>
    with_triton_call signature call_args [N := N] in triton {
      zero = tl.full([], tl.cast(0, tl.int64), dtype=tl.int64)
      group = tl.full([], tl.cast(B, tl.int64), dtype=tl.int64)
      one = tl.full([], tl.cast(1, tl.int64), dtype=tl.int64)
      last = N + group
      for row in range(tl.cast(N, tl.int64), tl.cast(last, tl.int64), tl.cast(1, tl.int64)):
        value = tl.load(X)
        tl.store(Y, value.to(tl.float8e4nv))
    }
  | .f8e5 =>
    with_triton_call signature call_args [N := N] in triton {
      zero = tl.full([], tl.cast(0, tl.int64), dtype=tl.int64)
      group = tl.full([], tl.cast(B, tl.int64), dtype=tl.int64)
      one = tl.full([], tl.cast(1, tl.int64), dtype=tl.int64)
      last = N + group
      for row in range(tl.cast(N, tl.int64), tl.cast(last, tl.int64), tl.cast(1, tl.int64)):
        value = tl.load(X)
        tl.store(Y, value.to(tl.float8e5))
    }
end VeriTile.Tests.FloatDTypeConstantReference
