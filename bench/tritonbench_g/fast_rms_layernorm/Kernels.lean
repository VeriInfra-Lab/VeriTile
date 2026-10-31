import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

/-! Complete readable FP32 sources. Runtime epsilon is an explicit scalar ABI
parameter. The preserved candidate commutes the integer X row-stride product;
all floating operations, casts, branches and stores retain their order. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm
open VeriTile.Triton

structure Call where
  rows : Nat
  cols : Nat
  block : Nat
  positive : 0<cols
  covered : cols≤block
  blockLimit : block≤65536
  gemma : Bool
  eps : ℝ

def Call.cells (h : Call) : Nat := h.rows*h.cols
namespace Kernels
noncomputable section

def regularRuntimeSource (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y,r]
  tl.param eps : tl.float32
  tl.region Y = tl.float32, X = tl.float32, W = tl.float32, r = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx * $(r_row_stride)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets * $(W_row_stride), mask=mask, other=0)
  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + eps)
  tl.store(r, inv_var.to(tl.float32))
  normed = X_row * inv_var
  normed = normed.to(tl.float32)
  output = normed * W_row
  tl.store(Y + col_offsets, output.to(tl.float32), mask=mask)
}

def gemmaRuntimeSource (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y,r]
  tl.param eps : tl.float32
  tl.region Y = tl.float32, X = tl.float32, W = tl.float32, r = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx * $(r_row_stride)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + eps)
  tl.store(r, inv_var.to(tl.float32))
  normed = X_row * inv_var
  output = normed * (W_row + 1.0)
  tl.store(Y + col_offsets, output.to(tl.float32), mask=mask)
}

def regularRuntimeSourceIndexCommuted (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y,r]
  tl.param eps : tl.float32
  tl.region Y = tl.float32, X = tl.float32, W = tl.float32, r = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y += row_idx * $(Y_row_stride)
  X += $(X_row_stride) * row_idx
  r += row_idx * $(r_row_stride)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets * $(W_row_stride), mask=mask, other=0)
  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + eps)
  tl.store(r, inv_var.to(tl.float32))
  normed = X_row * inv_var
  normed = normed.to(tl.float32)
  output = normed * W_row
  tl.store(Y + col_offsets, output.to(tl.float32), mask=mask)
}

def gemmaRuntimeSourceIndexCommuted (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y,r]
  tl.param eps : tl.float32
  tl.region Y = tl.float32, X = tl.float32, W = tl.float32, r = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y += row_idx * $(Y_row_stride)
  X += $(X_row_stride) * row_idx
  r += row_idx * $(r_row_stride)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + eps)
  tl.store(r, inv_var.to(tl.float32))
  normed = X_row * inv_var
  output = normed * (W_row + 1.0)
  tl.store(Y + col_offsets, output.to(tl.float32), mask=mask)
}

/-- Explicit call binding for the runtime source used by the Real judgment.
The assignment supplies the caller's scalar argument before the unchanged JIT body. -/
def regularSource (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : ComputeKernel :=
  (regularRuntimeSource Y X W r YR XR WR RR N B).withScalarArguments
    [ScalarArgument.fp32 "eps" eps]

/-- The same runtime ABI binding for the Gemma constexpr branch. -/
def gemmaSource (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : ComputeKernel :=
  (gemmaRuntimeSource Y X W r YR XR WR RR N B).withScalarArguments
    [ScalarArgument.fp32 "eps" eps]

def forwardRuntime (h : Call) : ComputeKernel :=
  if h.gemma then gemmaRuntimeSource "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block
  else regularRuntimeSource "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block

def inputGradientSource (dY X W r dW : RegionName) (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) (GEMMA : Bool) : ComputeKernel :=
      triton {
  tl.ports inputs=[dY,X,W,r], outputs=[dY]
  tl.region dY = tl.float32, X = tl.float32, W = tl.float32, r = tl.float32, dW = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx * $(r_row_stride)
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  if $((GEMMA : Bool)) {
    dY_W = dY_row * (W_row + 1.0)
  } else {
    dY_W = dY_row * W_row
  }
  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output.to(tl.float32), mask=mask)
}

def forward (h : Call) : ComputeKernel :=
  if h.gemma then gemmaSource "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps
  else regularSource "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps

def backward (h : Call) : ComputeKernel :=
  inputGradientSource "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps h.gemma

def weightBackward (h : Call) (K : Nat) : ComputeKernel :=
  let dW : RegionName := "dw"
  let dY : RegionName := "dx"
  let X : RegionName := "x"
  let r : RegionName := "r"
  triton {
    tl.ports inputs=["dx","x","r"], outputs=["dw"]
    tl.region dW = tl.float32, dY = tl.float32, X = tl.float32, r = tl.float32
    col = tl.program_id(0) * $(K) + tl.arange(0, $(K))
    mask = col < $(h.cols)
    acc = tl.zeros([$(K)], dtype=tl.float32)
    for row in range($(h.rows)) {
      dy = tl.load(dY + row * $(h.cols) + col, mask=mask, other=0).to(tl.float32)
      x = tl.load(X + row * $(h.cols) + col, mask=mask, other=0).to(tl.float32)
      inv = tl.load(r + row).to(tl.float32)
      normed = x * inv
      if $((!h.gemma : Bool)) {
        normed = normed.to(tl.float32).to(tl.float32)
      }
      acc = acc + dy * normed
    }
    tl.store(dW + col, acc.to(tl.float32), mask=mask)
  }

def inputGradientSourceIndexCommuted (dY X W r dW : RegionName) (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) (GEMMA : Bool) : ComputeKernel :=
      triton {
  tl.ports inputs=[dY,X,W,r], outputs=[dY]
  tl.region dY = tl.float32, X = tl.float32, W = tl.float32, r = tl.float32, dW = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  X += $(X_row_stride) * row_idx
  r += row_idx * $(r_row_stride)
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  if $((GEMMA : Bool)) {
    dY_W = dY_row * (W_row + 1.0)
  } else {
    dY_W = dY_row * W_row
  }
  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output.to(tl.float32), mask=mask)
}

def forwardRuntimeIndexCommuted (h : Call) : ComputeKernel :=
  if h.gemma then gemmaRuntimeSourceIndexCommuted "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block
  else regularRuntimeSourceIndexCommuted "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block

def backwardIndexCommuted (h : Call) : ComputeKernel :=
  inputGradientSourceIndexCommuted "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps h.gemma

end
end Kernels
end VeriTile.Bench.Repairs.FastRmsLayernorm
