import VeriTile.Triton

namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
open VeriTile.Triton
structure Parameters where
  (input weight output : RegionName)
  (rows columns innerSize inputRow inputInner weightInner weightColumn outputRow outputColumn
    blockRows blockColumns blockInner groupRows : Nat)
def inner (p : Parameters) : Nat := p.innerSize
def steps (p : Parameters) : Nat := (inner p+p.blockInner-1)/p.blockInner

end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
