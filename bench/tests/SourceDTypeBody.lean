import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SourceDTypeBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32)]
  sourceDTypePointers := ["V"]
  readPorts := ["X"]
  writePorts := ["V"]

/-- The same copied body specializes from the caller's storage type. Typed
conditional constexpr expressions need no parentheses added by the user. -/
def source (X V : RegionName) (dtype : SourceDType) : ComputeKernel :=
  with_triton_signature signature source_dtype V := dtype in triton {
    value = tl.load(X)
    if V.dtype.element_ty.is_int():
        is_bool: tl.constexpr = V.dtype.element_ty.is_int1()
        width: tl.constexpr = 8 if is_bool else V.dtype.element_ty.primitive_bitwidth
        signed: tl.constexpr = is_bool or V.dtype.element_ty.is_int_signed()
        exponent: tl.constexpr = width - 1 if signed else width
        lower: tl.constexpr = -(1 << exponent) if signed else 0
        gap: tl.constexpr = (1 << (exponent - 24)) if exponent > 24 else 1
        upper: tl.constexpr = (1 << exponent) - gap
        if signed:
            value = tl.maximum(value, tl.full((), lower, tl.float32))
        else:
            value = tl.maximum(value, tl.full((), 0, tl.float32))
        value = tl.minimum(value, tl.full((), upper, tl.float32))
    tl.store(V, value)
  }

def fp32Expected (X V : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[V]
  tl.region X = tl.float32, V = tl.float32
  value = tl.load(X)
  tl.store(V, value)
}

theorem fp32_has_no_fabricated_cast (X V : RegionName) :
    source X V .fp32 = fp32Expected X V := rfl

def halfExpected (X V : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[V]
  tl.region X = tl.float32, V = tl.float16
  value = tl.load(X)
  tl.store(V, value.to(tl.float16))
}

theorem half_preserves_conversion (X V : RegionName) :
    source X V .fp16 = halfExpected X V := rfl

def signedExpected (X V : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[V]
  tl.region X = tl.float32, V = tl.int8
  value = tl.load(X)
  value = tl.maximum(value, tl.full([], -128, dtype=tl.float32))
  value = tl.minimum(value, tl.full([], 127, dtype=tl.float32))
  tl.store(V, value.to(tl.int8))
}

theorem signed_bounds_and_conversion (X V : RegionName) :
    source X V .i8 = signedExpected X V := rfl

def boolExpected (X V : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[V]
  tl.region X = tl.float32, V = tl.int1
  value = tl.load(X)
  value = tl.maximum(value, tl.full([], -128, dtype=tl.float32))
  value = tl.minimum(value, tl.full([], 127, dtype=tl.float32))
  tl.store(V, value.to(tl.int8) != $((0 : Int)))
}

theorem bool_keeps_byte_conversion (X V : RegionName) :
    source X V .boolean = boolExpected X V := rfl

/-- error: source dtype pointer requires its source_dtype invocation binding -/
#guard_msgs in
def missingBinding (X V : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton { tl.store(V, tl.load(X)) }

/-- error: source_dtype requires exactly the declared source dtype pointer -/
#guard_msgs in
def wrongPointer (X V : RegionName) (dtype : SourceDType) : ComputeKernel :=
  with_triton_signature signature source_dtype X := dtype in triton {
    tl.store(V, tl.load(X))
  }

/-- error: source dtype pointer cannot also have a fixed pointer dtype -/
#guard_msgs in
def conflictingType (X V : RegionName) (dtype : SourceDType) : ComputeKernel :=
  with_triton_signature
    ({signature with pointers := [("X",.fp32),("V",.fp32)]} : CallSignature)
    source_dtype V := dtype in triton { tl.store(V, tl.load(X)) }

/- Metadata must be rejected even before dtype-dead branches are removed. -/
/-- error: pointer types belong in the call signature, outside the pasted body -/
#guard_msgs in
def hiddenMetadata (X V : RegionName) (dtype : SourceDType) : ComputeKernel :=
  with_triton_signature signature source_dtype V := dtype in triton {
    if V.dtype.element_ty.is_int():
        tl.region V = tl.float32
    tl.store(V, tl.load(X))
  }

/-- error: dtype constexpr assignment under runtime control flow -/
#guard_msgs in
def dynamicConstant (X V : RegionName) (dtype : SourceDType) : ComputeKernel :=
  with_triton_signature signature source_dtype V := dtype in triton {
    value = tl.load(X)
    if value > 0:
        width: tl.constexpr = V.dtype.element_ty.primitive_bitwidth
    tl.store(V, value)
  }

/-- error: runtime assignment shadows a dtype constexpr -/
#guard_msgs in
def shadowedConstant (X V : RegionName) (dtype : SourceDType) : ComputeKernel :=
  with_triton_signature signature source_dtype V := dtype in triton {
    width: tl.constexpr = V.dtype.element_ty.primitive_bitwidth
    width = tl.program_id(0)
    tl.store(V, tl.load(X))
  }

#axiomsClean fp32_has_no_fabricated_cast
#axiomsClean half_preserves_conversion
#axiomsClean signed_bounds_and_conversion
#axiomsClean bool_keeps_byte_conversion
end VeriTile.Tests.SourceDTypeBody
