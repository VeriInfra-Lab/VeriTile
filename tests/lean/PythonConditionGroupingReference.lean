import VeriTile.Triton.DSL
namespace VeriTile.Tests.PythonConditionGroupingReference
open VeriTile.Triton
/-- Independently written, already grouped typed source. Runtime integer
inputs remain free, and both stores retain their actual FP32 value conversions. -/
def source (Y : RegionName) : ComputeKernel := triton {
  tl.call_signature
  tl.ports inputs=[], outputs=[Y]
  tl.region Y = tl.float32
  tl.param a : tl.int64
  tl.param b : tl.int64
  tl.param c : tl.int64
  if ((a > 0 and b > 1) and c > 2) {
    value = 11.0
  } else {
    value = 13.0
  }
  other = (17.0 if a > 0 else 19.0)
  tl.store(Y, value.to(tl.float32))
  tl.store(Y + 1, other.to(tl.float32))
}
end VeriTile.Tests.PythonConditionGroupingReference
