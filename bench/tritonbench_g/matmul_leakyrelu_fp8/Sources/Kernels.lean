import VeriTile.Triton

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.TypedSource
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def configuration : Nat × Nat × Nat × Nat × Nat × Nat := (128,256,64,8,2,8)
def activationEnabled (activation : String) : Bool := activation == "leaky_relu"
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.TypedSource
