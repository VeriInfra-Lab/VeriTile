import bench.repairs.tritonbench_g.swiglu_fwd.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SwigluFwd.SourcePreservation
open VeriTile.Triton

def previous_kernel (X Y OUT : RegionName) (sx sy so cols block : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[X, Y], outputs=[OUT]
    tl.region X = tl.float32, Y = tl.float32, OUT = tl.float32
    row = tl.program_id(0)
    start_col = tl.program_id(1) * $(block)
    X += row * $(sx)
    Y += row * $(sy)
    OUT += row * $(so)
    cols = start_col + tl.arange(0, $(block))
    x = tl.load(X + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < $(cols))
  }

def previous_indexCommutedKernel (X Y OUT : RegionName) (sx sy so cols block : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[X, Y], outputs=[OUT]
    tl.region X = tl.float32, Y = tl.float32, OUT = tl.float32
    row = tl.program_id(0)
    start_col = tl.program_id(1) * $(block)
    X += $(sx) * row
    Y += $(sy) * row
    OUT += $(so) * row
    cols = start_col + tl.arange(0, $(block))
    x = tl.load(X + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < $(cols))
  }


theorem kernel_preserved (X Y OUT : RegionName) (sx sy so cols block : Nat) : kernel X Y OUT sx sy so cols block = previous_kernel X Y OUT sx sy so cols block  := by
  have zero : (0. : ℝ) = (0.0 : ℝ) := by norm_num
  simp only [kernel, indexCommutedKernel, previous_kernel, previous_indexCommutedKernel, zero]
theorem indexCommutedKernel_preserved (X Y OUT : RegionName) (sx sy so cols block : Nat) : indexCommutedKernel X Y OUT sx sy so cols block = previous_indexCommutedKernel X Y OUT sx sy so cols block  := by
  have zero : (0. : ℝ) = (0.0 : ℝ) := by norm_num
  simp only [kernel, indexCommutedKernel, previous_kernel, previous_indexCommutedKernel, zero]

/-- Preserve the original public whole-function connections as well. -/
theorem original : kernel = sourceKernel .fp32 .fp32 .fp32 Bool.false := by
  funext X Y OUT sx sy so cols block
  exact (kernel_preserved X Y OUT sx sy so cols block).trans rfl

theorem indexCommuted : indexCommutedKernel = sourceKernel .fp32 .fp32 .fp32 Bool.true := by
  funext X Y OUT sx sy so cols block
  exact (indexCommutedKernel_preserved X Y OUT sx sy so cols block).trans rfl

#axiomsClean kernel_preserved
#axiomsClean indexCommutedKernel_preserved
end VeriTile.Bench.Repairs.SwigluFwd.SourcePreservation
