import bench.tritonbench_g.layer_norm_fwd.Sources.Mathematical_Kernels
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Contracts

/-! Symbolic complete-source fp32 checkpoint. Other storage profiles remain
separate obligations; the source macro retains their dtype parameters. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Triton

abbrev sourceType := Flags → Regions → Strides → Nat → Nat → ℝ → ℝ →
  (block : Nat) → Random.ArangeSizeDomain block → ComputeKernel

def original : sourceType :=
  fun f r stride M N eps p block hBlock =>
    (layerNormFwdSource%[tl.float32, tl.float32, tl.float32, tl.float32,
      tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, original])
    r.x r.y r.weight r.bias r.residual r.x1 r.weight1 r.bias1 r.y1
    r.residualOut r.rowscale r.seeds.cast r.dropoutMask.cast r.mean r.rstd
    stride.x stride.y stride.residual stride.residualOut stride.x1 stride.y1
    M N eps p f.rms block hBlock f.residual f.residualOut f.bias f.dropout
    f.dropoutMask f.rowscale f.x1 f.w1 f.b1

def candidate : sourceType :=
  fun f r stride M N eps p block hBlock =>
    (layerNormFwdSource%[tl.float32, tl.float32, tl.float32, tl.float32,
      tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, indexCommuted])
    r.x r.y r.weight r.bias r.residual r.x1 r.weight1 r.bias1 r.y1
    r.residualOut r.rowscale r.seeds.cast r.dropoutMask.cast r.mean r.rstd
    stride.x stride.y stride.residual stride.residualOut stride.x1 stride.y1
    M N eps p f.rms block hBlock f.residual f.residualOut f.bias f.dropout
    f.dropoutMask f.rowscale f.x1 f.w1 f.b1

end VeriTile.Bench.TritonBenchG.LayerNormFwd
