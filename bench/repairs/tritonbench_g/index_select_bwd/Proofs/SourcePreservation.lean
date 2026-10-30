import bench.repairs.tritonbench_g.index_select_bwd.Kernels
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural

/-! Previous full programs are proof-only preservation witnesses. -/
namespace VeriTile.Bench.Repairs.IndexSelectBwd.SourcePreservation
open VeriTile.Triton

def previous_kernel (p : Launch) : ComputeKernel :=
  let grad_source_ptr := p.destination
  let grad_output_ptr := p.incoming
  triton {
    tl.region grad_source_ptr = tl.float32, grad_output_ptr = tl.float32
    lane = tl.arange(0, $(p.block)).to(tl.int64)
    active = lane < $((p.count : Int))
    safe = tl.where(active, lane, $((0 : Int)))
    current = tl.load(grad_source_ptr + safe * $((p.stride : Int)), mask=active, other=0).to(tl.float32)
    incoming = tl.load(grad_output_ptr + safe, mask=active, other=0).to(tl.float32)
    result = current + incoming
    tl.store(grad_source_ptr + safe * $((p.stride : Int)), result, mask=active)
  }


/-- The old metadata collector missed both casted loads. The call signature
now lists both actual read ports; every executable statement is unchanged. -/
theorem kernel_preserved (p : Launch) :
    (kernel p).surfaceBody = (previous_kernel p).surfaceBody := rfl

theorem execution_preserved {A : Type} [Inhabited A] (M : FP.Structural.Algebra A)
    (p : Launch) (s : FP.Structural.State A) :
    FP.Structural.exec M (kernel p) s = FP.Structural.exec M (previous_kernel p) s := rfl

#axiomsClean kernel_preserved
#axiomsClean execution_preserved
end VeriTile.Bench.Repairs.IndexSelectBwd.SourcePreservation
