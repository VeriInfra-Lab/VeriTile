import bench.repairs.tritonbench_g.dropout_triton.Proofs.Region
import VeriTile.Triton.Memory.RegionalKernelSpec

namespace VeriTile.Bench.Repairs.DropoutTriton.IO
open VeriTile Triton
noncomputable section

/-- FP32 data, signed-int32 keep mask and FP32 output share one masked tile.
The source keeps these precisions; Real IO changes only their mathematical
interpretation. There are no private scratch writes. -/
def dropoutIO (n : Nat) (p : ℝ) : MixedMaskedKernelIO₂ where
  source := kernel n p
  projection := RegionProof.projection Bool.false n blockSize p
  in1 := "x"
  in2 := "keep"
  out := "out"
  type1 := .float
  type2 := .int
  outType := .float
  B1 := blockSize
  B2 := blockSize
  Bout := blockSize
  read1 := fun pid j => pid*blockSize+j.val
  read2 := fun pid j => pid*blockSize+j.val
  write := fun pid j => pid*blockSize+j.val
  mask1 := fun pid j => pid*blockSize+j.val<n
  mask2 := fun pid j => pid*blockSize+j.val<n
  outMask := fun pid j => pid*blockSize+j.val<n

end
end VeriTile.Bench.Repairs.DropoutTriton.IO
