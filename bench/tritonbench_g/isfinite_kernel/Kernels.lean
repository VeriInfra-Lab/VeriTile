import VeriTile.Triton.DSL.ImportedFiniteJitFunctions
import bench.tritonbench_g.isfinite_kernel.Contract

/-! Complete copied classifier helper and strided caller bodies. Source import
aliases, call dtypes, signed byte strides and allocation-relative pointer bases
are supplied outside the bodies. The existing candidate commutes only the
natural block-index product; both explicit pointer-element casts remain. -/
namespace VeriTile.Bench.Repairs.IsfiniteKernel
open VeriTile.Triton

def isfiniteSource : ImportedFiniteJitFunction :=
  with_triton_imports [_isfinited := "triton.language.extra.cuda.libdevice.isfinited",
    _finitef := "triton.language.extra.cuda.libdevice.finitef"] in with_triton_params (x) constexpr_args [] in triton {
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))
  }

def callSignature (precision : Precision) : CallSignature where
  pointers := [("in0_ptr",match precision with | .single => .fp32 | .double => .fp64),("out0_ptr",.boolean)]
  scalars := [("in0_stride",.integer),("out0_stride",.integer),("n",.integer)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["in0_ptr"]
  writePorts := ["out0_ptr"]

def sourceKernel (precision : Precision) (in0_ptr out0_ptr : RegionName)
    (in0_stride out0_stride : Int) (n BLOCK_SIZE : Nat) (inputOffset outputOffset : Int) : ComputeKernel :=
  with_triton_call (callSignature precision) call_args [in0_stride := in0_stride, out0_stride := out0_stride, n := n]
    signed_pointer_offsets [in0_ptr := inputOffset, out0_ptr := outputOffset]
    pure_functions [isfinite_func := isfiniteSource] in triton {
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + indices * tl.cast(in0_stride, tl.int64), mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + indices * tl.cast(out0_stride, tl.int64),
             out0.to(out0_ptr.type.element_ty), mask=mask)
  }

def candidateKernel (precision : Precision) (in0_ptr out0_ptr : RegionName)
    (in0_stride out0_stride : Int) (n BLOCK_SIZE : Nat) (inputOffset outputOffset : Int) : ComputeKernel :=
  with_triton_call (callSignature precision) call_args [in0_stride := in0_stride, out0_stride := out0_stride, n := n]
    signed_pointer_offsets [in0_ptr := inputOffset, out0_ptr := outputOffset]
    pure_functions [isfinite_func := isfiniteSource] in triton {
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + indices * tl.cast(in0_stride, tl.int64), mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + indices * tl.cast(out0_stride, tl.int64),
             out0.to(out0_ptr.type.element_ty), mask=mask)
  }

def source32 (input output : View) (n B : Nat) : ComputeKernel :=
  sourceKernel .single "x" "y" input.byteStride output.byteStride n B input.pointer output.pointer

def source64 (input output : View) (n B : Nat) : ComputeKernel :=
  sourceKernel .double "x" "y" input.byteStride output.byteStride n B input.pointer output.pointer

def candidate32 (input output : View) (n B : Nat) : ComputeKernel :=
  candidateKernel .single "x" "y" input.byteStride output.byteStride n B input.pointer output.pointer

def candidate64 (input output : View) (n B : Nat) : ComputeKernel :=
  candidateKernel .double "x" "y" input.byteStride output.byteStride n B input.pointer output.pointer

def source (precision : Precision) (input output : View) (n B : Nat) : ComputeKernel :=
  match precision with
  | .single => source32 input output n B
  | .double => source64 input output n B

def indexCommuted (precision : Precision) (input output : View) (n B : Nat) : ComputeKernel :=
  match precision with
  | .single => candidate32 input output n B
  | .double => candidate64 input output n B

end VeriTile.Bench.Repairs.IsfiniteKernel
