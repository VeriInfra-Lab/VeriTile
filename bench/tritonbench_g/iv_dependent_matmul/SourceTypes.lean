import VeriTile.Triton.Core.Types

namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul
open VeriTile.Triton
namespace FPValue
inductive Graph where | source | accumulator deriving DecidableEq, Repr
structure Parameters where
  (input weight output : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride blockRows blockColumns blockInner : Nat)
end FPValue
namespace Kernels
inductive Schedule where
  | preLoad | postLoad | mixed | two | three
  deriving DecidableEq, Repr

def Schedule.name : Schedule → String
  | .preLoad => "pre_load"
  | .postLoad => "post_load"
  | .mixed => "post_pre_mixed"
  | .two => "post_load_two_iters"
  | .three => "post_load_three_iters"

def Schedule.stages (s : Schedule) : Nat := if s = .three then 4 else 3

end Kernels
end VeriTile.Bench.TritonBenchG.IvDependentMatmul
