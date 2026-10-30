import VeriTile.Triton.DSL.SourceDType
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SourceDType
open VeriTile.Triton

/-- Dtype specialization must retain the load, NaN test and both clamps. -/
def source (X V : RegionName) (dtype : SourceDType) : ComputeKernel :=
  triton (dtype V := dtype) {
    tl.region X = tl.float32
    value = tl.load(X)
    if V.dtype.element_ty.is_int() {
      lower : tl.constexpr = -128
      upper : tl.constexpr = 127
      value = tl.where(value == value, value, 0.0)
      value = tl.minimum(tl.maximum(value, tl.full([], lower, dtype=tl.float32)), tl.full([], upper, dtype=tl.float32))
    }
    p = V + $(0)
    tl.store(p, value)
  }

def signedExpected (X V : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, V = tl.int8
  value = tl.load(X)
  value = tl.where(value == value, value, 0.0)
  value = tl.minimum(tl.maximum(value, tl.full([], -128, dtype=tl.float32)), tl.full([], 127, dtype=tl.float32))
  p = V + $(0)
  tl.store(p, (value).to(tl.int8))
}

theorem integer_keeps_clamps_and_cast (X V : RegionName) :
    source X V .i8 = signedExpected X V := rfl

def halfExpected (X V : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, V = tl.float16
  value = tl.load(X)
  p = V + $(0)
  tl.store(p, (value).to(tl.float16))
}

theorem half_keeps_store_rounding (X V : RegionName) :
    source X V .fp16 = halfExpected X V := rfl

/-- Both FP8 formats are floating branches and keep their actual store format. -/
def fp8E4Expected (X V : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, V = tl.float8e4nv
  value = tl.load(X)
  p = V + $(0)
  tl.store(p, value.to(tl.float8e4nv))
}
def fp8E5Expected (X V : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, V = tl.float8e5
  value = tl.load(X)
  p = V + $(0)
  tl.store(p, value.to(tl.float8e5))
}
theorem fp8_formats_are_distinct (X V : RegionName) :
    source X V .f8e4 = fp8E4Expected X V ∧
      source X V .f8e5 = fp8E5Expected X V := ⟨rfl,rfl⟩
#axiomsClean fp8_formats_are_distinct

def masked (X V : RegionName) : ComputeKernel := triton (dtype V := SourceDType.u8) {
  tl.region X = tl.float32
  tl.param n : tl.int64
  value = tl.load(X)
  if n > $((0 : Int)) {
    tl.store(V, value, mask=n > $((1 : Int)))
  } else {
    tl.store(V, value)
  }
}

def maskedExpected (X V : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, V = tl.uint8
  tl.param n : tl.int64
  value = tl.load(X)
  if n > $((0 : Int)) {
    tl.store(V, (value).to(tl.uint8), mask=n > $((1 : Int)))
  } else {
    tl.store(V, (value).to(tl.uint8))
  }
}

theorem both_arms_and_mask_keep_cast (X V : RegionName) : masked X V = maskedExpected X V := rfl

/-- error: runtime assignment shadows a dtype constexpr -/
#guard_msgs in
def rebind (V : RegionName) : ComputeKernel := triton (dtype V := SourceDType.i8) {
  width : tl.constexpr = V.dtype.element_ty.primitive_bitwidth
  width = tl.program_id(0)
}

/-- error: dtype constexpr assignment under runtime control flow -/
#guard_msgs in
def dynamicConstant (V : RegionName) : ComputeKernel := triton (dtype V := SourceDType.i8) {
  tl.param n : tl.int64
  if n > $((0 : Int)) {
    width : tl.constexpr = V.dtype.element_ty.primitive_bitwidth
  }
}

/-- error: dtype pointer aliases cannot be rebound under runtime control flow -/
#guard_msgs in
def branchAlias (V : RegionName) : ComputeKernel := triton (dtype V := SourceDType.i8) {
  tl.param n : tl.int64
  if n > $((0 : Int)) {
    p = V + $(0)
  }
}

/-- error: unsupported store control flow in dtype specialization -/
#guard_msgs in
def nestedStore (V : RegionName) : ComputeKernel := triton (dtype V := SourceDType.i8) {
  for i in range($(0), $(4), $(1)) {
    tl.store(V + i, $((0 : Int)))
  }
}

/-- error: tl.argmax: unsigned integer tiles of rank > 1 require an explicit axis -/
#guard_msgs in
def ambiguousUnsignedAxis : ComputeKernel := triton {
  x = tl.arange(0, 4)
  index = tl.argmax(x[:, None])
}

/-- error: tl.argmax: boolean tiles of rank > 1 require an explicit axis -/
#guard_msgs in
def ambiguousBoolAxis : ComputeKernel := triton {
  x = tl.arange(0, 4)
  values = x < $(2)
  index = tl.argmax(values[:, None])
}

/-- error: dtype pointer aliases cannot be rebound under runtime control flow -/
#guard_msgs in
def escapedLoopAlias (V : RegionName) : ComputeKernel := triton (dtype V := SourceDType.i8) {
  for i in range(4) {
    p = V + i
  }
  tl.store(p, $((0 : Int)))
}

#axiomsClean integer_keeps_clamps_and_cast
#axiomsClean half_keeps_store_rounding
#axiomsClean both_arms_and_mask_keep_cast
end VeriTile.Tests.SourceDType
