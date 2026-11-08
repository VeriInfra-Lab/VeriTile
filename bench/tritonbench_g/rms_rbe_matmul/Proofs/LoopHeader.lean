import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelModel
import bench.tritonbench_g.rms_matmul_rbe.Proofs.RangeHeader

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

/-- The literal and symbolic scalar interfaces share these integer arguments.
No floating value or materialization rule participates in the loop header. -/
def headerArguments (p : Parameters) : RmsMatmulRbeScalarDomain.Arguments :=
  ⟨"X","W","RMS","OUT",p.M,p.N,p.K,p.M*p.K,p.K,1,1,p.K,1,p.M*p.N,p.N,1,
    16,64,64,p.headDim,p.start,p.rotary⟩

def loopCode (fp8 : Bool) (p : Parameters) := ((kernel fp8 p).surfaceBody.drop 24).take 1

theorem loopCode_header (fp8 : Bool) (p : Parameters) :
    loopCode fp8 p=[.forRangeInt "step"
      (RmsMatmulRbeScalarDomain.startExpr (headerArguments p))
      (RmsMatmulRbeScalarDomain.countExpr (headerArguments p))
      (RmsMatmulRbeScalarDomain.strideExpr (headerArguments p)) (loop fp8 p)] := by
  cases fp8 <;> rfl

theorem loopCode_run {α : Type} [Inhabited α] (a : Operations α)
    (fp8 : Bool) (p : Parameters) (s : State α) :
    run (engine a) (loopCode fp8 p) s=
      rangeInt (engine a) "step" 0 (steps p) 1 (loop fp8 p) s := by
  rw [loopCode_header]
  have bounds:=RmsMatmulRbeScalarDomain.range_header (engine a) (headerArguments p) s
  simp only [run,step]
  rw [bounds.1,bounds.2.1,bounds.2.2]
  change (rangeInt (engine a) "step" 0 (steps p) 1 (loop fp8 p) s).bind some = _
  cases rangeInt (engine a) "step" 0 (steps p) 1 (loop fp8 p) s <;> rfl

#axiomsClean loopCode_header
#axiomsClean loopCode_run
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
