import bench.repairs.tritonbench_g.index_select_cat.Kernels
namespace VeriTile.Bench.Repairs.IndexSelectCat.LegacyPastedSurface
open VeriTile.Triton

def kernel : Layout → ComputeKernel :=
  fun (p : Layout) =>
    let output_ptr := p.output
    let source_ptr := p.source
    let index_ptr := p.index
    triton {
      tl.ports inputs=[source_ptr,index_ptr], outputs=[output_ptr]
      tl.region output_ptr = tl.float32, source_ptr = tl.float32, index_ptr = tl.int64
      pid0 = tl.program_id(axis=0)
      pid1 = tl.program_id(axis=1)
      indices = pid0 * $(p.BI) + tl.arange(0, $(p.BI))
      rows = tl.load(index_ptr + indices, mask=indices < $(p.rows), other=$((0 : Int)))
      cols = pid1 * $(p.BC) + tl.arange(0, $(p.BC))
      source_offsets = source_ptr + rows[:, None] * $(p.rowStride) + cols[None, :] * $(p.colStride)
      mask = (indices[:, None] < $(p.rows)) & (cols[None, :] < $(p.cols))
      output = tl.load(source_offsets, mask=mask)
      output_offsets = output_ptr + indices[:, None] * $(p.rowStride) + cols[None, :] * $(p.colStride)
      tl.store(output_offsets, output, mask=mask)
    }

def indexCommutedKernel : Layout → ComputeKernel :=
  fun (p : Layout) =>
    let output_ptr := p.output
    let source_ptr := p.source
    let index_ptr := p.index
    triton {
      tl.ports inputs=[source_ptr,index_ptr], outputs=[output_ptr]
      tl.region output_ptr = tl.float32, source_ptr = tl.float32, index_ptr = tl.int64
      pid0 = tl.program_id(axis=0)
      pid1 = tl.program_id(axis=1)
      indices = $(p.BI) * pid0 + tl.arange(0, $(p.BI))
      rows = tl.load(index_ptr + indices, mask=indices < $(p.rows), other=$((0 : Int)))
      cols = pid1 * $(p.BC) + tl.arange(0, $(p.BC))
      source_offsets = source_ptr + rows[:, None] * $(p.rowStride) + cols[None, :] * $(p.colStride)
      mask = (indices[:, None] < $(p.rows)) & (cols[None, :] < $(p.cols))
      output = tl.load(source_offsets, mask=mask)
      output_offsets = output_ptr + indices[:, None] * $(p.rowStride) + cols[None, :] * $(p.colStride)
      tl.store(output_offsets, output, mask=mask)
    }

end VeriTile.Bench.Repairs.IndexSelectCat.LegacyPastedSurface
