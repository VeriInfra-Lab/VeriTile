import VeriTile.Triton

namespace VeriTile.Bench.Repairs.RbeTritonTransform
open VeriTile.Triton
noncomputable section

structure Request where
  batches : Nat
  rows : Nat
  columns : Nat
  position : Int

def Request.size (r : Request) := r.batches*r.rows*r.columns
def Request.offset (r : Request) (b m c : Nat) := (b*r.rows+m)*r.columns+c

namespace Source
structure Parameters where
  (input output : RegionName)
  (rows columns inputBatchStride inputRowStride inputColumnStride
    outputBatchStride outputRowStride outputColumnStride : Nat)
  startToken : Int
  theta : ℝ
  (frequencyDimension blockRows blockColumns : Nat)

end Source

abbrev parameters (r : Request) (input output : RegionName) : Source.Parameters :=
  ⟨input,output,r.rows,r.columns,r.rows*r.columns,r.columns,1,
    r.rows*r.columns,r.columns,1,r.position,10000,128,2,1024⟩

end
end VeriTile.Bench.Repairs.RbeTritonTransform
