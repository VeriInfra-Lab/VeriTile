import bench.tests.PointerOffsets
namespace VeriTile.Tests.PointerOffsetRebindingBody
open VeriTile.Triton
open VeriTile.Tests.PointerOffsets

def source (X Y : RegionName) (xb : Int) (yb n B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [n := n] pointer_offsets [X := xb, Y := yb] in triton {
    before = tl.load(X)
    X = X + n
    X = X + 1
    for i in range(0, n):
        X = X + 1
    after = tl.load(X)
    tl.store(Y, after)
  }

def reference (X Y : RegionName) (xb : Int) (yb n B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float32, Y = tl.float32
  before = tl.load(X + $((xb : Int)))
  X = X + $((xb : Int)) + $(n)
  X = X + 1
  for i in range(0, $(n), $((1 : Nat))) {
    X = X + 1
  }
  after = tl.load(X)
  tl.store(Y + $(yb), after)
}

/-- Caller offset is used before and at the first assignment. The later
assignment and the loop, including zero trips, read the local pointer. -/
theorem complete (X Y : RegionName) (xb : Int) (yb n B : Nat) :
    source X Y xb yb n B = reference X Y xb yb n B := rfl

def conditional (X Y : RegionName) (xb n B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [n := n] pointer_offsets [X := xb] in triton {
    if tl.program_id(0) == 0:
        X = X + 1
    x = tl.load(X)
    tl.store(Y, x)
  }

def conditionalExpected (X Y : RegionName) (xb n B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float32, Y = tl.float32
  if tl.program_id(0) == 0 {
    X = X + $((xb : Nat)) + $(1)
    x = tl.load(X)
    tl.store(Y, x)
  } else {
    x = tl.load(X + $((xb : Nat)))
    tl.store(Y, x)
  }
}

/-- The first conditional pointer assignment consumes the caller offset only
on the assigned arm; the unassigned arm still reads from that entry offset. -/
theorem conditional_complete (X Y : RegionName) (xb n B : Nat) :
    conditional X Y xb n B = conditionalExpected X Y xb n B := rfl

#axiomsClean conditional_complete

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def augmented (X Y : RegionName) (xb n B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [n := n] pointer_offsets [X := xb] in triton {
    X += 1
    x = tl.load(X)
    tl.store(Y, x)
  }

#axiomsClean complete
end VeriTile.Tests.PointerOffsetRebindingBody
