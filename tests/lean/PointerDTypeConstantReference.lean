import VeriTile.Triton.DSL.SourceDType
namespace VeriTile.Tests.PointerDTypeConstantReference
open VeriTile.Triton
/-- Independent typed reference, compiled before the optional invocation adapter. -/
def reference (X Y : RegionName) : SourceDType → ComputeKernel
  | .fp32 => triton {
      tl.ports inputs=[X], outputs=[Y]
      tl.region X = tl.float32, Y = tl.float32
      value = tl.load(X, dtype=tl.float32)
      tl.store(Y, value)
    }
  | kind => triton (dtype Y := kind) {
    tl.ports inputs=[X], outputs=[Y]
    tl.region X = tl.float32
    value = tl.load(X, dtype=tl.float32)
    if Y.dtype.element_ty.is_int() {
      is_bool : tl.constexpr = Y.dtype.element_ty.is_int1()
      width : tl.constexpr = (8 if is_bool else Y.dtype.element_ty.primitive_bitwidth)
      signed : tl.constexpr = (is_bool or Y.dtype.element_ty.is_int_signed())
      exponent : tl.constexpr = ((width - 1) if signed else width)
      lower : tl.constexpr = (-((1 << exponent)) if signed else 0)
      gap : tl.constexpr = ((1 << (exponent - 24)) if (exponent > 24) else 1)
      upper : tl.constexpr = ((1 << exponent) - gap)
      value = tl.where(value == value, value, 0.0)
      value = tl.minimum(tl.maximum(value, tl.full([], lower, dtype=tl.float32)),
        tl.full([], upper, dtype=tl.float32))
    }
    tl.store(Y, value)
  }
end VeriTile.Tests.PointerDTypeConstantReference
