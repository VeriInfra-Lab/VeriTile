import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! Primitive DSL regression draft, requiring shared patches 01 and 02.
Helper/import binding and the complete pinned writer are separate next layers. -/
namespace VeriTile.Tests.IsfiniteDSL
open VeriTile.Triton

def source32 (input : RegionName) : ComputeKernel := triton {
  tl.region input = tl.float32
  in0 = tl.load(input + tl.arange(0, 4))
  out0 = tl.extra.cuda.libdevice.finitef(in0)
}

def source64 (input : RegionName) : ComputeKernel := triton {
  tl.region input = tl.float64
  in0 = tl.load(input + tl.arange(0, 4))
  out0 = tl.extra.cuda.libdevice.isfinited(in0)
}

def short32 (input : RegionName) : ComputeKernel := triton {
  tl.region input = tl.float32
  in0 = tl.load(input + tl.arange(0, 4))
  out0 = libdevice.finitef(in0)
}

def short64 (input : RegionName) : ComputeKernel := triton {
  tl.region input = tl.float64
  in0 = tl.load(input + tl.arange(0, 4))
  out0 = libdevice.isfinited(in0)
}

theorem source32_classifier (input : RegionName) :
    (source32 input).surfaceBody.getLast? = some
      (.assign .bool [4] "out0" (.isFinite .fp32
        (.compute (.alg .fp32 (.ref .real [4] "in0"))))) := rfl

theorem source64_classifier (input : RegionName) :
    (source64 input).surfaceBody.getLast? = some
      (.assign .bool [4] "out0" (.isFinite .fp64
        (.compute (.alg .fp64 (.ref .real [4] "in0"))))) := rfl

theorem qualified_spellings (input : RegionName) :
    short32 input = source32 input ∧ short64 input = source64 input := ⟨rfl, rfl⟩

def castBfloat16 (input : RegionName) : ComputeKernel := triton {
  tl.region input = tl.bfloat16
  in0 = tl.load(input + tl.arange(0, 4))
  out0 = tl.extra.cuda.libdevice.finitef(in0.to(tl.float32))
}

/-- The helper's explicit bf16-to-fp32 conversion stays under classification. -/
theorem bfloat16_cast_retained (input : RegionName) :
    (castBfloat16 input).surfaceBody.getLast? = some
      (.assign .bool [4] "out0" (.isFinite .fp32
        (.compute (.alg .fp32 (.castFloat .bf16 .real (.ref .bf16 [4] "in0")))))) := rfl

/-- The materialized Bool register is a normal Bool operand for later source
logic; it must not acquire a floating/opaque-register precision annotation. -/
def subsequentBool (input : RegionName) : ComputeKernel := triton {
  tl.region input = tl.float32
  in0 = tl.load(input + tl.arange(0, 4))
  out0 = tl.extra.cuda.libdevice.finitef(in0)
  opposite = tl.logical_not(out0)
}

theorem bool_register_retained (input : RegionName) :
    (subsequentBool input).surfaceBody.getLast? = some
      (.assign .bool [4] "opposite" (.alg (.boolNot (.ref .bool [4] "out0")))) := rfl

theorem source32_not_real (input : RegionName) :
    (source32 input).toAlgorithm? =
      .error (.requiresComputeSemantics "libdevice finite classification") := rfl

theorem source64_not_real (input : RegionName) :
    (source64 input).toAlgorithm? =
      .error (.requiresComputeSemantics "libdevice finite classification") := rfl

/-- error: finite classification requires its explicit fp32/fp64 operand precision -/
#guard_msgs in
example : ComputeKernel := triton {
  value = (1.0).to(tl.float64)
  finite = tl.extra.cuda.libdevice.finitef(value)
}

/-- error: finite classification requires its explicit fp32/fp64 operand precision -/
#guard_msgs in
example : ComputeKernel := triton {
  value = (1.0).to(tl.float32)
  finite = tl.extra.cuda.libdevice.isfinited(value)
}

-- A mathematical Real operand supplies no fp64 storage evidence.
/-- error: finite classification requires its explicit fp32/fp64 operand precision -/
#guard_msgs in
example : ComputeKernel := triton {
  value = 1.0
  finite = tl.extra.cuda.libdevice.isfinited(value)
}

#axiomsClean source32_classifier
#axiomsClean source64_classifier
#axiomsClean qualified_spellings
#axiomsClean bfloat16_cast_retained
#axiomsClean bool_register_retained
#axiomsClean source32_not_real
#axiomsClean source64_not_real
end VeriTile.Tests.IsfiniteDSL
