import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.Sources.Mathematical_Kernels

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 48000
namespace Original
abbrev Parameters := VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.Parameters

abbrev kernel := VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.kernel
end Original

/-- Complete backward source: original operation order, 29 corrected decimal
coefficient occurrences, three old-zero gradient loads and three stores.
The Python launch explicitly selects enable_fp_fusion=False in both phases. -/
def backwardSource (coord_ptr coord_grad_ptr sph_grad_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride upstream_row_stride upstream_col_stride : Nat) : ComputeKernel := backwardPastedKernel coord_ptr coord_grad_ptr sph_grad_ptr block_size coord_numel output_numel col_offset output_stride upstream_row_stride upstream_col_stride

/-- Share left-associated rounded powers. The numerical operation tree of every
used power is unchanged; recomputation is removed without reassociation. -/
def forwardPowersSource (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) : ComputeKernel := forwardPowersPastedKernel coord_ptr output_ptr block_size coord_numel output_numel col_offset output_stride
def backwardPowersSource (coord_ptr coord_grad_ptr sph_grad_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride upstream_row_stride upstream_col_stride : Nat) : ComputeKernel := backwardPowersPastedKernel coord_ptr coord_grad_ptr sph_grad_ptr block_size coord_numel output_numel col_offset output_stride upstream_row_stride upstream_col_stride

def coordinateRegion : RegionName := "coordinates"
def outputRegion : RegionName := "output"
def upstreamRegion : RegionName := "upstream"
def gradientRegion : RegionName := "gradient"

structure Parameters where
  rows : Nat
  blockSize : Nat
  blockPositive : 0 < blockSize
  outputWidth : Nat
  columnOffset : Nat
  upstreamRowStride : Nat
  upstreamColStride : Nat

def forwardParameters (p : Parameters) : Original.Parameters :=
  ⟨coordinateRegion,outputRegion,p.blockSize,3*p.rows,p.outputWidth*p.rows,p.columnOffset,p.outputWidth⟩

def forwardKernel (p : Parameters) := Original.kernel (forwardParameters p)

def backwardKernel (p : Parameters) :=
  backwardSource coordinateRegion gradientRegion upstreamRegion
    p.blockSize (3*p.rows) (p.outputWidth*p.rows) p.columnOffset p.outputWidth
    p.upstreamRowStride p.upstreamColStride

def forwardPowersKernel (p : Parameters) :=
  forwardPowersSource coordinateRegion outputRegion p.blockSize (3*p.rows)
    (p.outputWidth*p.rows) p.columnOffset p.outputWidth

def backwardPowersKernel (p : Parameters) :=
  backwardPowersSource coordinateRegion gradientRegion upstreamRegion
    p.blockSize (3*p.rows) (p.outputWidth*p.rows) p.columnOffset p.outputWidth
    p.upstreamRowStride p.upstreamColStride

end VeriTile.Bench.Repairs.FifthOrderSphHarmonics
