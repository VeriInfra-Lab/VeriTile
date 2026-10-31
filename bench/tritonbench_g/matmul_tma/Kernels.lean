import bench.tritonbench_g.matmul_tma.Contract

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2

noncomputable abbrev params := StaticChecked.params
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
open VeriTile.Triton FPValue Host
noncomputable section

/-- The FP32 output already has the dot accumulator's format. Remove its
identity conversion; the FP16 branch keeps the original complete source. -/
def sourceDirectOutput (p : Parameters) (half : Bool) : ComputeKernel :=
  let dtype : FloatDType := .fp16
  if half then source p Bool.true else triton {
    a_block_ptr = tl.make_block_ptr(base=$(p.a), shape=($(p.rows), $(p.inner)),
      strides=($(p.aRow), $(p.aInner)), offsets=($(0), $(0)),
      block_shape=($(p.blockRows), $(p.blockInner)), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=$(p.b), shape=($(p.inner), $(p.columns)),
      strides=($(p.bInner), $(p.bColumn)), offsets=($(0), $(0)),
      block_shape=($(p.blockInner), $(p.blockColumns)), order=(0, 1))
    c_block_ptr = tl.make_block_ptr(base=$(p.c), shape=($(p.rows), $(p.columns)),
      strides=($(p.cRow), $(p.cColumn)), offsets=($(0), $(0)),
      block_shape=($(p.blockRows), $(p.blockColumns)), order=(1, 0))
    a = tl.load(a_block_ptr, dtype=DTYPE)
    b = tl.load(b_block_ptr, dtype=DTYPE)
    c = tl.dot(a, b)
    tl.store(c_block_ptr, c)
  }

def kernelDirectOutput (h : Shape) (f : Flags) : ComputeKernel :=
  sourceDirectOutput (params h f) f.half
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
