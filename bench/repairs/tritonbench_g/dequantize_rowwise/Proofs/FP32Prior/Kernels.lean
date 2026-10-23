import VeriTile.Triton.Source
import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior
open VeriTile.Triton

/-- Full repaired source at the selected FP32/FP32 to FP16 route. The native
pid/arange widenings have the same natural-index projection; Native.lean
separately checks every fixed-width intermediate and active byte pointer.
The two multiplications remain left-associated and the store cast is explicit.
The old INT8 source and all old public theorems are unchanged. -/
def source (x scale out : RegionName) (inverse : ℝ) (n cols width : Nat) : ComputeKernel :=
  triton {
    tl.region x = tl.float32, scale = tl.float32, out = tl.float16
    pid = tl.program_id(axis=0)
    block_start = pid * $(cols)
    arange = tl.arange(0, $(width))
    offsets = block_start + arange
    row_mask = arange < $(cols)
    x_value = tl.load(x + offsets, mask=row_mask)
    max_val = tl.load(scale + pid)
    output = max_val * x_value * $((inverse : ℝ))
    tl.store(out + offsets, (output).to(tl.float16), mask=row_mask)
  }

def xRegion : RegionName := "x"
def scaleRegion : RegionName := "scale"
def outRegion : RegionName := "out"

structure Parameters where
  rows : Nat
  cols : Nat
  width : Nat
  covers : cols ≤ width

noncomputable def kernel (p : Parameters) := source xRegion scaleRegion outRegion (1/127) (p.rows*p.cols) p.cols p.width

end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior
