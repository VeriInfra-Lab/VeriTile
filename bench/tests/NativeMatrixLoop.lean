import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NativeMatrixLoop
open VeriTile.Triton

def profile : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["S","N"]

def source (X Y : RegionName) (S N : Nat) : ComputeKernel :=
  with_triton_signature profile in
  triton {
    rows = tl.arange(0, S)
    cols = tl.arange(0, S)
    offsets = rows[:, None] * S + cols[None, :]
    x = tl.load(X + offsets)
    u = tl.zeros([S], dtype=tl.float32)
    for i in range(N):
        u = tl.sum(x + u[:, None], axis=1)
    y = tl.dot(x, x, input_precision="ieee")
    tl.store(Y + offsets, y)
  }

def expanded (X Y : RegionName) (S N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  rows = tl.arange(0, $(S))
  cols = tl.arange(0, $(S))
  offsets = rows[:, None] * $(S) + cols[None, :]
  x = tl.load(X + offsets)
  u = tl.zeros([$(S)], dtype=tl.float32)
  tl.for i in $(N) {
    u = tl.sum(x + u[:, None], axis=1)
  }
  y = tl.dot(x, x, input_precision="ieee")
  tl.store(Y + offsets, y)
}

/-- Broadcasting preserves FP32 loop state. A known FP32 dot result is stored
without inventing a redundant conversion; the dot precision option remains. -/
theorem complete_source (X Y : RegionName) (S N : Nat) :
    source X Y S N = expanded X Y S N := rfl

/-- error: loop-carried value requires a stable, known source dtype -/
#guard_msgs in
def mixedLoop (X Y : RegionName) (S N : Nat) : ComputeKernel :=
  with_triton_signature profile in
  triton {
    rows = tl.arange(0, S)
    cols = tl.arange(0, S)
    offsets = rows[:, None] * S + cols[None, :]
    x = tl.load(X + offsets)
    u = tl.zeros([S], dtype=tl.float32)
    for i in range(N):
        u = tl.sum(x.to(tl.float64) + u[:, None], axis=1)
    tl.store(Y + rows, u)
  }

#axiomsClean complete_source
end VeriTile.Tests.NativeMatrixLoop
