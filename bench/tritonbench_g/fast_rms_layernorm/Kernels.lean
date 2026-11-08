import VeriTile.Triton
import VeriTile.Triton.DSL.ChainedSourceCasts
import VeriTile.Meta.StatementAudit

/-! Complete readable FP32 sources. All bodies are copied from the Python functions. Runtime epsilon is an explicit scalar ABI
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

def forwardSignature : CallSignature where
  pointers := [("Y",.fp32),("X",.fp32),("W",.fp32),("r",.fp32)]
  scalars := [("Y_row_stride",.integer),("X_row_stride",.integer),
    ("W_row_stride",.integer),("r_row_stride",.integer),("n_cols",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X","W"]
  writePorts := ["Y","r"]

def backwardSignature : CallSignature where
  pointers := [("dY",.fp32),("X",.fp32),("W",.fp32),("r",.fp32),("dW",.fp32)]
  scalars := [("dY_row_stride",.integer),("X_row_stride",.integer),
    ("W_row_stride",.integer),("r_row_stride",.integer),("dW_row_stride",.integer),
    ("n_cols",.integer),("eps",.fp32)]
  constexpr := ["GEMMA","BLOCK_SIZE"]
  readPorts := ["dY","X","W","r"]
  writePorts := ["dY"]

def weightSignature : CallSignature where
  pointers := [("dW",.fp32),("dY",.fp32),("X",.fp32),("r",.fp32)]
  scalars := [("n_rows",.integer),("n_cols",.integer)]
  constexpr := ["GEMMA","BLOCK_SIZE"]
  readPorts := ["dY","X","r"]
  writePorts := ["dW"]

def regularRuntimeKernel (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,
    X_row_stride := X_row_stride, W_row_stride := W_row_stride,
    r_row_stride := r_row_stride, n_cols := n_cols] in triton {
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets * W_row_stride, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)
  }

def gemmaRuntimeKernel (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,
    X_row_stride := X_row_stride, W_row_stride := W_row_stride,
    r_row_stride := r_row_stride, n_cols := n_cols] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask=mask)
  }

def regularRuntimeSourceIndexCommutedKernel (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,
    X_row_stride := X_row_stride, W_row_stride := W_row_stride,
    r_row_stride := r_row_stride, n_cols := n_cols] in triton {
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += X_row_stride * row_idx
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets * W_row_stride, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)
  }

def gemmaRuntimeSourceIndexCommutedKernel (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,
    X_row_stride := X_row_stride, W_row_stride := W_row_stride,
    r_row_stride := r_row_stride, n_cols := n_cols] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += X_row_stride * row_idx
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask=mask)
  }

def inputGradientKernel (dY X W r dW : RegionName)
    (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (GEMMA : Bool) : ComputeKernel :=
  with_triton_call backwardSignature call_args [dY_row_stride := dY_row_stride,
    X_row_stride := X_row_stride, W_row_stride := W_row_stride,
    r_row_stride := r_row_stride, dW_row_stride := dW_row_stride, n_cols := n_cols] in triton {
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask=mask, other=0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA: dY_W = dY_row * (W_row + 1.0)
    else:     dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask=mask)
  }

def inputGradientSourceIndexCommutedKernel (dY X W r dW : RegionName)
    (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (GEMMA : Bool) : ComputeKernel :=
  with_triton_call backwardSignature call_args [dY_row_stride := dY_row_stride,
    X_row_stride := X_row_stride, W_row_stride := W_row_stride,
    r_row_stride := r_row_stride, dW_row_stride := dW_row_stride, n_cols := n_cols] in triton {
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += X_row_stride * row_idx
    r  += row_idx *  r_row_stride

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask=mask, other=0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA: dY_W = dY_row * (W_row + 1.0)
    else:     dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask=mask)
  }

def runtimeWeightKernel (dW dY X r : RegionName)
    (n_rows n_cols BLOCK_SIZE : Nat) (GEMMA : Bool) : ComputeKernel :=
  with_triton_call weightSignature call_args [n_rows := n_rows, n_cols := n_cols] in triton {
    col = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols
    acc = tl.full((BLOCK_SIZE,), 0, tl.float32)
    for row in range(tl.cast(n_rows, tl.int64)):
        dy = tl.load(dY + row * n_cols + col, mask=mask, other=0).to(tl.float32)
        x = tl.load(X + row * n_cols + col, mask=mask, other=0).to(tl.float32)
        inv = tl.load(r + row).to(tl.float32)
        normed = x * inv
        if not GEMMA:
            normed = normed.to(dW.dtype.element_ty).to(tl.float32)
        acc = acc + dy * normed
    tl.store(dW + col, acc, mask=mask)
  }

/-- Explicit call binding for the runtime source used by the Real judgment.
The assignment supplies the caller's scalar argument before the unchanged JIT body. -/
def regularSource (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : ComputeKernel :=
  (regularRuntimeKernel Y X W r YR XR WR RR N B).withScalarArguments
    [ScalarArgument.fp32 "eps" eps]

/-- The same runtime ABI binding for the Gemma constexpr branch. -/
def gemmaSource (Y X W r : RegionName)
    (YR XR WR RR N B : Nat) (eps : ℝ) : ComputeKernel :=
  (gemmaRuntimeKernel Y X W r YR XR WR RR N B).withScalarArguments
    [ScalarArgument.fp32 "eps" eps]

def forwardRuntime (h : Call) : ComputeKernel :=
  if h.gemma then gemmaRuntimeKernel "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block
  else regularRuntimeKernel "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block


def forward (h : Call) : ComputeKernel :=
  if h.gemma then gemmaSource "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps
  else regularSource "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps

def backward (h : Call) : ComputeKernel :=
  inputGradientKernel "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps h.gemma


def forwardRuntimeIndexCommuted (h : Call) : ComputeKernel :=
  if h.gemma then gemmaRuntimeSourceIndexCommutedKernel "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block
  else regularRuntimeSourceIndexCommutedKernel "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block

def backwardIndexCommuted (h : Call) : ComputeKernel :=
  inputGradientSourceIndexCommutedKernel "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps h.gemma

def weightBackward (h : Call) (K : Nat) : ComputeKernel :=
  runtimeWeightKernel "dw" "dx" "x" "r" h.rows h.cols K h.gemma

end
end Kernels
end VeriTile.Bench.Repairs.FastRmsLayernorm
