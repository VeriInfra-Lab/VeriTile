import VeriTile.Triton.DSL
/-! The shared selected FP32 source and its genuine integer-index candidate. -/
namespace VeriTile.Bench.Repairs.KldivCompute.SelectedSource
open VeriTile.Triton
def original (X Y O : RegionName) (n block : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X,Y], outputs=[O]
  tl.region X = tl.float32, Y = tl.float32, O = tl.float32
  pid = tl.program_id(0)
  block_start = pid * $(block)
  offsets = block_start + tl.arange(0, $(block))
  mask = offsets < $(n)
  x = tl.load(X + offsets, mask=mask)
  y = tl.load(Y + offsets, mask=mask)
  output = x * tl.log(x / y)
  tl.store(O + offsets, output, mask=mask)
}

def commuted (X Y O : RegionName) (n block : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X,Y], outputs=[O]
  tl.region X = tl.float32, Y = tl.float32, O = tl.float32
  pid = tl.program_id(0)
  block_start = $(block) * pid
  offsets = block_start + tl.arange(0, $(block))
  mask = offsets < $(n)
  x = tl.load(X + offsets, mask=mask)
  y = tl.load(Y + offsets, mask=mask)
  output = x * tl.log(x / y)
  tl.store(O + offsets, output, mask=mask)
}

end VeriTile.Bench.Repairs.KldivCompute.SelectedSource
