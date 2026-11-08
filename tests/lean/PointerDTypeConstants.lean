import tests.lean.PointerDTypeConstantReference
import VeriTile.Triton.DSL.PointerDTypeConstants
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PointerDTypeConstants
open VeriTile.Triton

def signature (kind : SourceDType) : CallSignature where
  pointers := [("X",.fp32),("Y",kind)]
  scalars := []
  constexpr := []
  readPorts := ["X"]
  writePorts := ["Y"]

def source (X Y : RegionName) (kind : SourceDType) : ComputeKernel :=
  with_triton_signature (signature kind) in triton {
    value = tl.load(X)
    if Y.dtype.element_ty.is_int():
        is_bool: tl.constexpr = Y.dtype.element_ty.is_int1()
        width: tl.constexpr = 8 if is_bool else Y.dtype.element_ty.primitive_bitwidth
        signed: tl.constexpr = is_bool or Y.dtype.element_ty.is_int_signed()
        exponent: tl.constexpr = width - 1 if signed else width
        lower: tl.constexpr = -(1 << exponent) if signed else 0
        gap: tl.constexpr = (1 << (exponent - 24)) if exponent > 24 else 1
        upper: tl.constexpr = (1 << exponent) - gap
        value = tl.where(value == value, value, 0.0)
        value = tl.minimum(tl.maximum(value, tl.full((), lower, tl.float32)),
            tl.full((), upper, tl.float32))
    tl.store(Y, value)
  }

/-- Every storage dtype keeps the complete typed clamp and output program. -/
theorem complete_program : source=PointerDTypeConstantReference.reference := by
  funext X Y kind
  cases kind <;> rfl
/-- A literal dtype must not cause unrelated pointer types to be specialized. -/
def knownFp32 (X Y : RegionName) : ComputeKernel :=
  with_triton_signature (signature .fp32) in triton {
    value = tl.load(X)
    if Y.dtype.element_ty.is_int():
      value = value.to(tl.float32)
    tl.store(Y, value)
  }
theorem knownFp32_program (X Y : RegionName) :
    knownFp32 X Y=PointerDTypeConstantReference.reference X Y .fp32 := by rfl

/-- error: a queried pointer dtype requires an immutable pointer parameter -/
#guard_msgs (error) in
example (X Y : RegionName) (kind : SourceDType) : ComputeKernel :=
  with_triton_signature (signature kind) in triton {
    Y = X
    if Y.dtype.element_ty.is_int():
      value = tl.load(X)
      tl.store(Y, value)
  }

/-- error: dtype constexpr assignment under runtime control flow -/
#guard_msgs (error) in
example (X Y : RegionName) (kind : SourceDType) (flag : Bool) : ComputeKernel :=
  with_triton_signature (signature kind) in triton {
    if Y.dtype.element_ty.is_int():
      if flag:
        width: tl.constexpr = 8
    tl.store(Y, tl.load(X))
  }

/-- error: unsupported pointer dtype constexpr property -/
#guard_msgs (error) in
example (X Y : RegionName) (kind : SourceDType) : ComputeKernel :=
  with_triton_signature (signature kind) in triton {
    if Y.dtype.element_ty.unknown_property():
      value = tl.load(X)
      tl.store(Y, value)
  }

#axiomsClean complete_program
#axiomsClean knownFp32_program
end VeriTile.Tests.PointerDTypeConstants
