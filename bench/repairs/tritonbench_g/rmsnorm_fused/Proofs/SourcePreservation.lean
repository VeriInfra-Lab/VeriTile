import bench.repairs.tritonbench_g.rmsnorm_fused.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RmsnormFused.SourcePreservation
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

def previous_source (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  Y += row * $(N)
  X += row * $(N)
  _var = tl.zeros([$(B)], dtype=tl.float32)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + eps)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }
}

def previous_addressCommutedSource (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  tl.param eps : tl.float32
  row = tl.program_id(0).to(tl.int64)
  Y += $(N) * row
  X += row * $(N)
  _var = tl.zeros([$(B)], dtype=tl.float32)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + eps)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }
}


theorem source_preserved (X Y W : RegionName) (N B : Nat) :
    source X Y W N B = previous_source X Y W N B := rfl

theorem candidate_preserved (X Y W : RegionName) (N B : Nat) :
    addressCommutedSource X Y W N B = previous_addressCommutedSource X Y W N B := rfl

#axiomsClean source_preserved
#axiomsClean candidate_preserved
end VeriTile.Bench.Repairs.RmsnormFused.SourcePreservation
