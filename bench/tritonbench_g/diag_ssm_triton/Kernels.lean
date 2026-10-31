import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Complete selected source quotations for the four diagonal SSM kernels and
their existing integer-index candidates. Real storage is FP32; complex64 uses
the original interleaved FP32 components. Recurrences, reverse-time branches,
gradient signs, masks, simultaneous complex updates and stores are unchanged.
The existing source-profile names refer directly to these shared quotations.
The signed range bridge stays in Proofs/NativeBinding, outside FP semantics. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
open Lean VeriTile.Triton

def blockSize : Nat := 128

def forwardTyped :=
  fun (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      triton {
  tl.ports inputs=[s_ptr, x_ptr, lambda_ptr], outputs=[y_ptr]
  tl.region s_ptr = tl.float32, x_ptr = tl.float32, lambda_ptr = tl.float32, y_ptr = tl.float32
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  tl.for t in $(length) {
    offsets = t * $(batch_size) * $(dim) + col_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    s = s * Lambda + x
    tl.store(y_ptr + offsets, s, mask=mask)
  }
}

def backwardTyped :=
  fun (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      triton {
  tl.ports inputs=[s_ptr, lambda_ptr, y_ptr, grad_y_ptr], outputs=[grad_s_ptr, grad_x_ptr, grad_lambda_ptr]
  tl.region s_ptr = tl.float32, lambda_ptr = tl.float32, y_ptr = tl.float32, grad_s_ptr = tl.float32, grad_x_ptr = tl.float32, grad_lambda_ptr = tl.float32, grad_y_ptr = tl.float32
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  grad_s = tl.zeros_like(Lambda)
  grad_Lambda = tl.zeros_like(Lambda)
  tl.for i in $(length) {
    t = $(length) - 1 - i
    offsets = t * $(batch_size) * $(dim) + col_offsets
    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    if t > 0 {
      s = tl.load( y_ptr + offsets - $(batch_size) * $(dim), mask=mask, other=0)
    } else {
      s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    }
    grad_s = grad_y + grad_s
    grad_x = grad_s
    grad_Lambda += grad_s * s
    grad_s = grad_s * Lambda
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets, grad_s, mask=mask)
  tl.store(grad_lambda_ptr + col_offsets, grad_Lambda, mask=mask)
}

def complexForwardTyped :=
  fun (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      triton {
  tl.ports inputs=[s_ptr, x_ptr, lambda_ptr], outputs=[y_ptr]
  tl.region s_ptr = tl.float32, x_ptr = tl.float32, y_ptr = tl.float32, lambda_ptr = tl.float32
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  s_real = tl.load(s_ptr + col_offsets * 2, mask=mask, other=0)
  s_imag = tl.load(s_ptr + col_offsets * 2 + 1, mask=mask, other=0)
  lambda_real = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2, mask=mask, other=0)
  lambda_imag = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2 + 1, mask=mask, other=0)
  tl.for t in $(length) {
    offsets = (t * $(batch_size) * $(dim) + col_offsets) * 2
    x_real = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_imag = tl.load(x_ptr + offsets + 1, mask=mask, other=0)
    new_s_real = s_real * lambda_real - s_imag * lambda_imag + x_real
    new_s_imag = s_real * lambda_imag + s_imag * lambda_real + x_imag
    tl.store(y_ptr + offsets, new_s_real, mask=mask)
    tl.store(y_ptr + offsets + 1, new_s_imag, mask=mask)
    s_real, s_imag = new_s_real, new_s_imag
  }
}

def complexBackwardTyped :=
  fun (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      triton {
  tl.ports inputs=[s_ptr, lambda_ptr, y_ptr, grad_y_ptr], outputs=[grad_s_ptr, grad_x_ptr, grad_lambda_ptr]
  tl.region s_ptr = tl.float32, lambda_ptr = tl.float32, y_ptr = tl.float32, grad_s_ptr = tl.float32, grad_x_ptr = tl.float32, grad_lambda_ptr = tl.float32, grad_y_ptr = tl.float32
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  lambda_real = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2, mask=mask, other=0)
  lambda_imag = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2 + 1, mask=mask, other=0)
  grad_s_real = tl.zeros_like(lambda_real)
  grad_s_imag = tl.zeros_like(lambda_imag)
  grad_lambda_real = tl.zeros_like(lambda_real)
  grad_lambda_imag = tl.zeros_like(lambda_imag)
  tl.for i in $(length) {
    t = $(length) - 1 - i
    offsets = (t * $(batch_size) * $(dim) + col_offsets) * 2
    grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    grad_y_imag = -tl.load( grad_y_ptr + offsets + 1, mask=mask, other=0)
    if t > 0 {
      s_real = tl.load( y_ptr + offsets - 2 * $(batch_size) * $(dim), mask=mask, other=0)
      s_imag = tl.load( y_ptr + offsets - 2 * $(batch_size) * $(dim) + 1, mask=mask, other=0)
    } else {
      s_real = tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0)
      s_imag = tl.load( s_ptr + 2 * col_offsets + 1, mask=mask, other=0)
    }
    grad_s_real = grad_y_real + grad_s_real
    grad_s_imag = grad_y_imag + grad_s_imag
    grad_x_real = grad_s_real
    grad_x_imag = grad_s_imag
    grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
    grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
    grad_s_real = grad_x_real * lambda_real - grad_x_imag * lambda_imag
    grad_s_imag = grad_x_real * lambda_imag + grad_x_imag * lambda_real
    tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
    tl.store(grad_x_ptr + offsets + 1, -grad_x_imag, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets * 2, grad_s_real, mask=mask)
  tl.store(grad_s_ptr + col_offsets * 2 + 1, -grad_s_imag, mask=mask)
  tl.store( grad_lambda_ptr + col_offsets * 2, grad_lambda_real, mask=mask)
  tl.store( grad_lambda_ptr + col_offsets * 2 + 1, -grad_lambda_imag, mask=mask)
}

def forwardIndexCommuted :=
  fun (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      triton {
  tl.ports inputs=[s_ptr, x_ptr, lambda_ptr], outputs=[y_ptr]
  tl.region s_ptr = tl.float32, x_ptr = tl.float32, lambda_ptr = tl.float32, y_ptr = tl.float32
  col_idx = $(BLOCK_SIZE) * tl.program_id(0)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  tl.for t in $(length) {
    offsets = t * $(batch_size) * $(dim) + col_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    s = s * Lambda + x
    tl.store(y_ptr + offsets, s, mask=mask)
  }
}

def backwardIndexCommuted :=
  fun (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      triton {
  tl.ports inputs=[s_ptr, lambda_ptr, y_ptr, grad_y_ptr], outputs=[grad_s_ptr, grad_x_ptr, grad_lambda_ptr]
  tl.region s_ptr = tl.float32, lambda_ptr = tl.float32, y_ptr = tl.float32, grad_s_ptr = tl.float32, grad_x_ptr = tl.float32, grad_lambda_ptr = tl.float32, grad_y_ptr = tl.float32
  col_idx = $(BLOCK_SIZE) * tl.program_id(0)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  grad_s = tl.zeros_like(Lambda)
  grad_Lambda = tl.zeros_like(Lambda)
  tl.for i in $(length) {
    t = $(length) - 1 - i
    offsets = t * $(batch_size) * $(dim) + col_offsets
    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    if t > 0 {
      s = tl.load( y_ptr + offsets - $(batch_size) * $(dim), mask=mask, other=0)
    } else {
      s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    }
    grad_s = grad_y + grad_s
    grad_x = grad_s
    grad_Lambda += grad_s * s
    grad_s = grad_s * Lambda
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets, grad_s, mask=mask)
  tl.store(grad_lambda_ptr + col_offsets, grad_Lambda, mask=mask)
}

def complexForwardIndexCommuted :=
  fun (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      triton {
  tl.ports inputs=[s_ptr, x_ptr, lambda_ptr], outputs=[y_ptr]
  tl.region s_ptr = tl.float32, x_ptr = tl.float32, y_ptr = tl.float32, lambda_ptr = tl.float32
  col_idx = $(BLOCK_SIZE) * tl.program_id(0)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  s_real = tl.load(s_ptr + col_offsets * 2, mask=mask, other=0)
  s_imag = tl.load(s_ptr + col_offsets * 2 + 1, mask=mask, other=0)
  lambda_real = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2, mask=mask, other=0)
  lambda_imag = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2 + 1, mask=mask, other=0)
  tl.for t in $(length) {
    offsets = (t * $(batch_size) * $(dim) + col_offsets) * 2
    x_real = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_imag = tl.load(x_ptr + offsets + 1, mask=mask, other=0)
    new_s_real = s_real * lambda_real - s_imag * lambda_imag + x_real
    new_s_imag = s_real * lambda_imag + s_imag * lambda_real + x_imag
    tl.store(y_ptr + offsets, new_s_real, mask=mask)
    tl.store(y_ptr + offsets + 1, new_s_imag, mask=mask)
    s_real, s_imag = new_s_real, new_s_imag
  }
}

def complexBackwardIndexCommuted :=
  fun (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      triton {
  tl.ports inputs=[s_ptr, lambda_ptr, y_ptr, grad_y_ptr], outputs=[grad_s_ptr, grad_x_ptr, grad_lambda_ptr]
  tl.region s_ptr = tl.float32, lambda_ptr = tl.float32, y_ptr = tl.float32, grad_s_ptr = tl.float32, grad_x_ptr = tl.float32, grad_lambda_ptr = tl.float32, grad_y_ptr = tl.float32
  col_idx = $(BLOCK_SIZE) * tl.program_id(0)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  lambda_real = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2, mask=mask, other=0)
  lambda_imag = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2 + 1, mask=mask, other=0)
  grad_s_real = tl.zeros_like(lambda_real)
  grad_s_imag = tl.zeros_like(lambda_imag)
  grad_lambda_real = tl.zeros_like(lambda_real)
  grad_lambda_imag = tl.zeros_like(lambda_imag)
  tl.for i in $(length) {
    t = $(length) - 1 - i
    offsets = (t * $(batch_size) * $(dim) + col_offsets) * 2
    grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    grad_y_imag = -tl.load( grad_y_ptr + offsets + 1, mask=mask, other=0)
    if t > 0 {
      s_real = tl.load( y_ptr + offsets - 2 * $(batch_size) * $(dim), mask=mask, other=0)
      s_imag = tl.load( y_ptr + offsets - 2 * $(batch_size) * $(dim) + 1, mask=mask, other=0)
    } else {
      s_real = tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0)
      s_imag = tl.load( s_ptr + 2 * col_offsets + 1, mask=mask, other=0)
    }
    grad_s_real = grad_y_real + grad_s_real
    grad_s_imag = grad_y_imag + grad_s_imag
    grad_x_real = grad_s_real
    grad_x_imag = grad_s_imag
    grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
    grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
    grad_s_real = grad_x_real * lambda_real - grad_x_imag * lambda_imag
    grad_s_imag = grad_x_real * lambda_imag + grad_x_imag * lambda_real
    tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
    tl.store(grad_x_ptr + offsets + 1, -grad_x_imag, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets * 2, grad_s_real, mask=mask)
  tl.store(grad_s_ptr + col_offsets * 2 + 1, -grad_s_imag, mask=mask)
  tl.store( grad_lambda_ptr + col_offsets * 2, grad_lambda_real, mask=mask)
  tl.store( grad_lambda_ptr + col_offsets * 2 + 1, -grad_lambda_imag, mask=mask)
}

end VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
