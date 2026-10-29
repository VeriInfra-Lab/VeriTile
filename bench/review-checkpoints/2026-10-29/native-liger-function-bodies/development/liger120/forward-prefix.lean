import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature
open VeriTile.Triton
namespace LigerForwardProbe
def forwardSignature : CallSignature where
  typedFP32Storage := Bool.true
  pointers := [("Y_ptr",.fp32),("X_ptr",.fp32),("W_ptr",.fp32),("B_ptr",.fp32),("Mean_ptr",.fp32),("RSTD_ptr",.fp32)]
  scalars := [("Y_row_stride",.integer),("X_row_stride",.integer),("W_row_stride",.integer),("B_row_stride",.integer),("Mean_row_stride",.integer),("RSTD_row_stride",.integer),("n_cols",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X_ptr", "W_ptr", "B_ptr"]
  writePorts := ["Y_ptr", "Mean_ptr", "RSTD_ptr"]

def probe1
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
  }

def probe2
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
  }

def probe3
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
  }

def probe4
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
  }

def probe5
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
  }

def probe6
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
  }

def probe7
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
  }

def probe8
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols
  }

def probe9
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
  }

def probe10
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
  }

def probe11
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
  }

def probe12
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride
  }

def probe13
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
  }

def probe14
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
  }

def probe15
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)
  }

def probe16
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
  }

def probe17
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum(tl.where(mask, (X_row - mean) * (X_row - mean), 0.0), axis=0) / n_cols
  }

def probe18
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum(tl.where(mask, (X_row - mean) * (X_row - mean), 0.0), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
  }

def probe19
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum(tl.where(mask, (X_row - mean) * (X_row - mean), 0.0), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
  }

def probe20
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum(tl.where(mask, (X_row - mean) * (X_row - mean), 0.0), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)
  }

def probe21
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum(tl.where(mask, (X_row - mean) * (X_row - mean), 0.0), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)

    Y_row = (X_row - mean) * rstd * W_row + B_row
  }

def probe22
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := 
  with_triton_call forwardSignature call_args [Y_row_stride := Y_row_stride,X_row_stride := X_row_stride,W_row_stride := W_row_stride,B_row_stride := B_row_stride,Mean_row_stride := Mean_row_stride,RSTD_row_stride := RSTD_row_stride,n_cols := n_cols,eps := eps] in triton {
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum(tl.where(mask, (X_row - mean) * (X_row - mean), 0.0), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)

    Y_row = (X_row - mean) * rstd * W_row + B_row

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)
  }

end LigerForwardProbe
