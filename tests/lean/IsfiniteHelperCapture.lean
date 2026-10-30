import VeriTile.Triton.DSL

/-! Negative controls for finite helper bindings and lexical scope.
These deliberately exercise local elaborator bindings, not Python hash checks. -/
namespace VeriTile.Tests.IsfiniteHelperCapture
open VeriTile.Triton

/-- error: finite source binding captures an outer Lean parameter or local binding -/
#guard_msgs in
def capturedHelper (isfinite_func : Nat) : ComputeKernel := triton {
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_helper isfinite_func(x) {
    return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
  }
}

/-- error: finite source binding captures an outer Lean parameter or local binding -/
#guard_msgs in
def capturedAlias (_finitef : Nat) : ComputeKernel := triton {
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
}

/-- error: finite source binding must be fresh and unambiguous -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_helper isfinite_func(_finitef) {
    return (_isfinited(_finitef) if _finitef.dtype.is_fp64() else _finitef(_finitef.to(tl.float32)))
  }
}

/-- error: finite helper and extern names cannot be rebound, aliased, or used outside direct calls -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_helper isfinite_func(x) {
    return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
  }
  for isfinite_func in range($(0), $(1)) {
    value = 1.0
  }
}

/-- error: finite source declarations must precede the writer body at top level -/
#guard_msgs in
example : ComputeKernel := triton {
  if true {
    tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  }
}

/-- error: finite helper resolved calls are internal syntax -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_helper isfinite_func(x) {
    return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
  }
  value = isfinite_func(tl.__finite_helper_resolved(1.0))
}

end VeriTile.Tests.IsfiniteHelperCapture
