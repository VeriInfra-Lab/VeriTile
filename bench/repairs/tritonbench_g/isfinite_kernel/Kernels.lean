import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.isfinite_kernel.Contract

/-! Complete unchanged Python helper and both strided writer bodies. Source
imports, FP32/FP64 pointer types, signed byte-view origins and strides are
supplied by the caller. The existing IO uses byte-address regions; its width
and allocation refinement remains in the storage/host proofs. -/
namespace VeriTile.Bench.Repairs.IsfiniteKernel
open VeriTile.Triton

def finiteFunction : SourceFunction :=
  triton_function isfinite_func(x) in triton {
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))
  }

def callSignature32 : CallSignature where
  pointers := [("in0_ptr",.fp32),("out0_ptr",.boolean)]
  scalars := [("in0_stride",.integer),("out0_stride",.integer),("n",.integer)]
  constexpr := ["BLOCK_SIZE"]
  functions := [finiteFunction]
  finiteImports := [("_isfinited","triton.language.extra.cuda.libdevice.isfinited"),
    ("_finitef","triton.language.extra.cuda.libdevice.finitef")]

def source32 (input output : View) (n B : Nat) : ComputeKernel :=
  let in0_ptr : RegionName := "x"
  let out0_ptr : RegionName := "y"
  let BLOCK_SIZE := B
  with_triton_call callSignature32 call_args [
    in0_stride := input.byteStride, out0_stride := output.byteStride, n := n]
    pointer_offsets [in0_ptr := input.pointer, out0_ptr := output.pointer] in triton {
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + indices * tl.cast(in0_stride, tl.int64), mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + indices * tl.cast(out0_stride, tl.int64),
             out0.to(out0_ptr.type.element_ty), mask=mask)
  }

def candidate32 (input output : View) (n B : Nat) : ComputeKernel :=
  let in0_ptr : RegionName := "x"
  let out0_ptr : RegionName := "y"
  let BLOCK_SIZE := B
  with_triton_call callSignature32 call_args [
    in0_stride := input.byteStride, out0_stride := output.byteStride, n := n]
    pointer_offsets [in0_ptr := input.pointer, out0_ptr := output.pointer] in triton {
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + indices * tl.cast(in0_stride, tl.int64), mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + indices * tl.cast(out0_stride, tl.int64),
             out0.to(out0_ptr.type.element_ty), mask=mask)
  }

def callSignature64 : CallSignature where
  pointers := [("in0_ptr",.fp64),("out0_ptr",.boolean)]
  scalars := [("in0_stride",.integer),("out0_stride",.integer),("n",.integer)]
  constexpr := ["BLOCK_SIZE"]
  functions := [finiteFunction]
  finiteImports := [("_isfinited","triton.language.extra.cuda.libdevice.isfinited"),
    ("_finitef","triton.language.extra.cuda.libdevice.finitef")]

def source64 (input output : View) (n B : Nat) : ComputeKernel :=
  let in0_ptr : RegionName := "x"
  let out0_ptr : RegionName := "y"
  let BLOCK_SIZE := B
  with_triton_call callSignature64 call_args [
    in0_stride := input.byteStride, out0_stride := output.byteStride, n := n]
    pointer_offsets [in0_ptr := input.pointer, out0_ptr := output.pointer] in triton {
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + indices * tl.cast(in0_stride, tl.int64), mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + indices * tl.cast(out0_stride, tl.int64),
             out0.to(out0_ptr.type.element_ty), mask=mask)
  }

def candidate64 (input output : View) (n B : Nat) : ComputeKernel :=
  let in0_ptr : RegionName := "x"
  let out0_ptr : RegionName := "y"
  let BLOCK_SIZE := B
  with_triton_call callSignature64 call_args [
    in0_stride := input.byteStride, out0_stride := output.byteStride, n := n]
    pointer_offsets [in0_ptr := input.pointer, out0_ptr := output.pointer] in triton {
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + indices * tl.cast(in0_stride, tl.int64), mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + indices * tl.cast(out0_stride, tl.int64),
             out0.to(out0_ptr.type.element_ty), mask=mask)
  }

def source (precision : Precision) (input output : View) (n B : Nat) : ComputeKernel :=
  match precision with
  | .single => source32 input output n B
  | .double => source64 input output n B

def indexCommuted (precision : Precision) (input output : View) (n B : Nat) : ComputeKernel :=
  match precision with
  | .single => candidate32 input output n B
  | .double => candidate64 input output n B

end VeriTile.Bench.Repairs.IsfiniteKernel
