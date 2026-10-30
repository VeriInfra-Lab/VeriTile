import VeriTile.Triton.DSL
import bench.tritonbench_g.isfinite_kernel.Contract

namespace VeriTile.Bench.Repairs.IsfiniteKernel
open VeriTile.Triton

/-- Readable direct strided source, with the original dtype-dispatched helper.
Region indices are byte addresses: the complete signed base+index*byteStride
is converted once, at the memory boundary. `View` records that interpretation.
The candidate commutes only the natural block-index multiplication. -/
def source32 (input output : View) (n B : Nat) : ComputeKernel :=
  let in0_ptr : RegionName := "x"
  let out0_ptr : RegionName := "y"
  triton {
    tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
    tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
    tl.source_helper isfinite_func(x) {
      return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
    }
    tl.region in0_ptr = tl.float32, out0_ptr = tl.int1
    pid = tl.program_id(0)
    offsets = pid * $(B) + tl.arange(0, $(B))
    mask = offsets < $(n)
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + ($((input.pointer : Int)) + indices * $((input.byteStride : Int))),
      mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + ($((output.pointer : Int)) + indices * $((output.byteStride : Int))),
      out0.to(out0_ptr.type.element_ty), mask=mask)
  }

def source64 (input output : View) (n B : Nat) : ComputeKernel :=
  let in0_ptr : RegionName := "x"
  let out0_ptr : RegionName := "y"
  triton {
    tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
    tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
    tl.source_helper isfinite_func(x) {
      return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
    }
    tl.region in0_ptr = tl.float64, out0_ptr = tl.int1
    pid = tl.program_id(0)
    offsets = pid * $(B) + tl.arange(0, $(B))
    mask = offsets < $(n)
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + ($((input.pointer : Int)) + indices * $((input.byteStride : Int))),
      mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + ($((output.pointer : Int)) + indices * $((output.byteStride : Int))),
      out0.to(out0_ptr.type.element_ty), mask=mask)
  }

def candidate32 (input output : View) (n B : Nat) : ComputeKernel :=
  let in0_ptr : RegionName := "x"
  let out0_ptr : RegionName := "y"
  triton {
    tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
    tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
    tl.source_helper isfinite_func(x) {
      return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
    }
    tl.region in0_ptr = tl.float32, out0_ptr = tl.int1
    pid = tl.program_id(0)
    offsets = $(B) * pid + tl.arange(0, $(B))
    mask = offsets < $(n)
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + ($((input.pointer : Int)) + indices * $((input.byteStride : Int))),
      mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + ($((output.pointer : Int)) + indices * $((output.byteStride : Int))),
      out0.to(out0_ptr.type.element_ty), mask=mask)
  }

def candidate64 (input output : View) (n B : Nat) : ComputeKernel :=
  let in0_ptr : RegionName := "x"
  let out0_ptr : RegionName := "y"
  triton {
    tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
    tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
    tl.source_helper isfinite_func(x) {
      return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
    }
    tl.region in0_ptr = tl.float64, out0_ptr = tl.int1
    pid = tl.program_id(0)
    offsets = $(B) * pid + tl.arange(0, $(B))
    mask = offsets < $(n)
    indices = offsets.to(tl.int64)
    in0 = tl.load(in0_ptr + ($((input.pointer : Int)) + indices * $((input.byteStride : Int))),
      mask=mask).to(in0_ptr.type.element_ty)
    out0 = isfinite_func(in0)
    tl.store(out0_ptr + ($((output.pointer : Int)) + indices * $((output.byteStride : Int))),
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
