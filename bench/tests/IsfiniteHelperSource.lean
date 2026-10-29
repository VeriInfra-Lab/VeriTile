import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! Source-helper dispatch and rejection regressions. -/
namespace VeriTile.Tests.IsfiniteHelperSource
open VeriTile.Triton Lean

syntax "finiteHelperSource%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(finiteHelperSource%[$storage:tritonDType]) =>
    `(fun (input : RegionName) => triton {
      tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
      tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
      tl.source_helper isfinite_func(x) {
        return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
      }
      tl.region input = $storage
      in0 = tl.load(input + tl.arange(0, 4))
      out0 = isfinite_func(in0)
    })

def source32 := finiteHelperSource%[tl.float32]
def source64 := finiteHelperSource%[tl.float64]
def sourceBfloat16 := finiteHelperSource%[tl.bfloat16]

theorem fp32_dispatch_retains_cast (input : RegionName) :
    (source32 input).surfaceBody.getLast? = some
      (.assign .bool [4] "out0" (.isFinite .fp32
        (.compute (.alg .fp32 (.castFloat .real .real (.ref .real [4] "in0")))))) := rfl

theorem fp64_dispatch_has_no_narrowing (input : RegionName) :
    (source64 input).surfaceBody.getLast? = some
      (.assign .bool [4] "out0" (.isFinite .fp64
        (.compute (.alg .fp64 (.ref .real [4] "in0"))))) := rfl

theorem bfloat16_dispatch_retains_widening (input : RegionName) :
    (sourceBfloat16 input).surfaceBody.getLast? = some
      (.assign .bool [4] "out0" (.isFinite .fp32
        (.compute (.alg .fp32 (.castFloat .bf16 .real (.ref .bf16 [4] "in0")))))) := rfl

def typeSpelling (input : RegionName) := triton {
  tl.region input = tl.float64
  value = tl.load(input).to(input.type.element_ty)
}

def dtypeSpelling (input : RegionName) := triton {
  tl.region input = tl.float64
  value = tl.load(input).to(input.dtype.element_ty)
}

theorem pointer_type_matches_dtype (input : RegionName) :
    typeSpelling input = dtypeSpelling input := rfl

def boolIdentity := triton {
  value = true
  copied = value.to(tl.int1)
}

theorem bool_identity_cast : boolIdentity.surfaceBody.getLast? = some
    (.assign .bool [] "copied" (.alg (.ref .bool [] "value"))) := rfl

/-- error: finite helper dtype.is_fp64 requires a known concrete floating source dtype -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_helper isfinite_func(x) {
    return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
  }
  value = 1.0
  out0 = isfinite_func(value)
}

/-- error: finite helper call lacks the matching declared source extern alias -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_helper isfinite_func(x) {
    return (_finitef(x) if x.dtype.is_fp64() else _isfinited(x.to(tl.float32)))
  }
}

/-- error: finite helper dtype predicate must inspect its declared parameter -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_helper isfinite_func(x) {
    return (_isfinited(x) if y.dtype.is_fp64() else _finitef(x.to(tl.float32)))
  }
}

/-- error: finite helper narrow branch requires its explicit float32 cast -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_helper isfinite_func(x) {
    return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x))
  }
}

/-- error: finite helper and extern names cannot be rebound, aliased, or used outside direct calls -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_helper isfinite_func(x) {
    return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
  }
  isfinite_func = 1.0
}

/-- error: finite helper resolved calls are internal syntax -/
#guard_msgs in
example : ComputeKernel := triton {
  value = tl.__finite_helper_resolved(1.0)
}

#axiomsClean fp32_dispatch_retains_cast
#axiomsClean fp64_dispatch_has_no_narrowing
#axiomsClean bfloat16_dispatch_retains_widening
#axiomsClean pointer_type_matches_dtype
#axiomsClean bool_identity_cast
end VeriTile.Tests.IsfiniteHelperSource
