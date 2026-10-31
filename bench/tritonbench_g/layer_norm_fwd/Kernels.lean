import bench.tritonbench_g.layer_norm_fwd.Sources.Kernels

/-! The positive-tile repair retains the complete ordinary JIT sources, including all optional streams, random
operations, statistics, masks and destination conversion sites. -/
namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Source
open VeriTile.Triton
abbrev Flags := VeriTile.Bench.Repairs.LayerNormFwd.Source.Flags
abbrev Regions := VeriTile.Bench.Repairs.LayerNormFwd.Source.Regions
abbrev Strides := VeriTile.Bench.Repairs.LayerNormFwd.Source.Strides
abbrev sourceType := VeriTile.Bench.Repairs.LayerNormFwd.Source.sourceType

/-- Complete existing FP32 source, retaining every optional stream and store. -/
def kernel : sourceType :=
  fun f r stride M N eps p block hBlock =>
    VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawOriginal
    r.x r.y r.weight r.bias r.residual r.x1 r.weight1 r.bias1 r.y1
    r.residualOut r.rowscale r.seeds.cast r.dropoutMask.cast r.mean r.rstd
    stride.x stride.y stride.residual stride.residualOut stride.x1 stride.y1
    M N eps p f.rms block hBlock f.residual f.residualOut f.bias f.dropout
    f.dropoutMask f.rowscale f.x1 f.w1 f.b1

/-- Existing candidate: only the initial integer X row/stride product is commuted. -/
def indexCommuted : sourceType :=
  fun f r stride M N eps p block hBlock =>
    VeriTile.Bench.Repairs.LayerNormFwdSelectedSource.rawIndexCommuted
    r.x r.y r.weight r.bias r.residual r.x1 r.weight1 r.bias1 r.y1
    r.residualOut r.rowscale r.seeds.cast r.dropoutMask.cast r.mean r.rstd
    stride.x stride.y stride.residual stride.residualOut stride.x1 stride.y1
    M N eps p f.rms block hBlock f.residual f.residualOut f.bias f.dropout
    f.dropoutMask f.rowscale f.x1 f.w1 f.b1
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Source
