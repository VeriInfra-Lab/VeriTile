import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.AliasedNestedLoadBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp32)]
  scalars := [("scale",.fp32)]
  constexpr := ["N"]

def source (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    p = X + 0
    __veritile_source_load_0 = tl.load(Y)
    for i in range(0, N):
        x = tl.load(p).to(tl.float32) * scale
        y = tl.load(Y) + x
        tl.store(Y, y)
        p += 1
  }

def explicitLoads (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    p = X + 0
    __veritile_source_load_0 = tl.load(Y)
    for i in range(0, N):
        __veritile_source_load_1 = tl.load(p)
        x = __veritile_source_load_1.to(tl.float32) * scale
        __veritile_source_load_2 = tl.load(Y)
        y = __veritile_source_load_2 + x
        tl.store(Y, y)
        p += 1
  }

/-- Preserve the FP16 read before the explicit FP32 cast, followed by the
second FP32 read; do not capture the user's existing temporary-shaped name. -/
theorem complete (X Y : RegionName) (N : Nat) :
    source X Y N = explicitLoads X Y N := rfl

/-- error: nested floating load requires supported eager arithmetic operands -/
#guard_msgs in
def lazyRead (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    p = X + 0
    x = (tl.load(p) if N > 0 else 0.0) * scale
    tl.store(Y, x)
  }

/-- error: nested load source precision is unsupported by the current frontend -/
#guard_msgs in
def integerDestination (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    p = X + 0
    x = tl.load(p).to(tl.int64) * scale
    tl.store(Y, x)
  }

#axiomsClean complete
end VeriTile.Tests.AliasedNestedLoadBody
