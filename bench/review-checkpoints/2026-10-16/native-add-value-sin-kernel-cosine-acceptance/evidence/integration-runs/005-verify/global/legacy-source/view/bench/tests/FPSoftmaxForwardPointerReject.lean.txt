import VeriTile.Triton.DSL

namespace VeriTile.Tests.ForwardPointerJoinNegative
open VeriTile.Triton

/-- error: pointer storage dtype is unknown on a bound branch: p -/
#guard_msgs in
def unknownBound (A B Y : RegionName) (flag : Bool) : ComputeKernel := triton {
  tl.region A = tl.float16, Y = tl.float16
  if flag { p = A + 0 } else { p = B + 0 }
  value = tl.load(p)
  tl.store(Y, (value).to(tl.float16))
}

/-- error: pointer storage dtype conflict across branches: p -/
#guard_msgs in
def nestedConflict (A B Y : RegionName) (outer innerLeft innerRight : Bool) : ComputeKernel := triton {
  tl.region A = tl.float16, B = tl.bfloat16, Y = tl.float16
  if outer {
    if innerLeft { p = A + 0 }
  } else {
    if innerRight { p = B + 0 }
  }
  value = tl.load(p)
  tl.store(Y, (value).to(tl.float16))
}

end VeriTile.Tests.ForwardPointerJoinNegative
