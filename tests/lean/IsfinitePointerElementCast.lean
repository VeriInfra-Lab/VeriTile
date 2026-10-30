import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! Focused, uncompiled regressions for the finite-helper cast prepass.
Root runs these against the isolated stage. -/
namespace VeriTile.Tests.IsfinitePointerElementCast
open Lean VeriTile.Triton

set_option hygiene false in
macro "pointerTypeCase%" "[" storage:tritonDType "]" " { " body:tritonStmt* " }" : term =>
  `(fun (input output : RegionName) => triton {
    tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
    tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
    tl.source_helper isfinite_func(x) {
      return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
    }
    tl.region input = tl.float32, output = $storage
    $body:tritonStmt*
  })

def widePointer := pointerTypeCase%[tl.float64] {
  value = tl.load(input)
  converted = value.to(output.type.element_ty)
}

def wideExplicit := pointerTypeCase%[tl.float64] {
  value = tl.load(input)
  converted = value.to(tl.float64)
}

theorem pointer_target_retains_explicit_conversion : widePointer = wideExplicit := rfl

theorem pointer_target_retains_fp64 (input output : RegionName) :
    (widePointer input output).surfaceBody.getLast? = some
      (.assign .real [] "converted" (.compute (.alg .fp64
        (.castFloat .real .real (.withPrecision .real .fp32
          (.ref .real [] "value")))))) := rfl

def boolPointer := pointerTypeCase%[tl.int1] {
  value = tl.load(input)
  finite = isfinite_func(value)
  converted = finite.to(output.type.element_ty)
}

def boolExplicit := pointerTypeCase%[tl.int1] {
  value = tl.load(input)
  finite = isfinite_func(value)
  converted = finite.to(tl.int1)
}

theorem bool_pointer_matches_explicit : boolPointer = boolExplicit := rfl

theorem bool_pointer_preserves_operand (input output : RegionName) :
    (boolPointer input output).surfaceBody.getLast? = some
      (.assign .bool [] "converted" (.alg (.ref .bool [] "finite"))) := rfl

/-- error: random source helper requires an explicit tl.source_helper binding -/
#guard_msgs in
example := pointerTypeCase%[tl.int1] {
  converted = foo.to(unknown_target)
}

/-- error: random source helper requires an explicit tl.source_helper binding -/
#guard_msgs in
example := pointerTypeCase%[tl.int1] {
  value = tl.load(input)
  converted = value.to(container.output.type.element_ty)
}

/-- error: pointer .type.element_ty cast requires known supported storage provenance -/
#guard_msgs in
example := pointerTypeCase%[tl.int1] {
  value = tl.load(input)
  converted = value.to(missing.type.element_ty)
}

/-- error: finite helper and extern names cannot be rebound, aliased, or used outside direct calls -/
#guard_msgs in
example := pointerTypeCase%[tl.int1] {
  converted = _finitef.to(output.type.element_ty)
}

/-- error: finite helper and extern names cannot be rebound, aliased, or used outside direct calls -/
#guard_msgs in
example := pointerTypeCase%[tl.int1] {
  converted = isfinite_func.to(output.type.element_ty)
}

/-- error: finite helper and extern names cannot be rebound, aliased, or used outside direct calls -/
#guard_msgs in
example := pointerTypeCase%[tl.int1] {
  value = tl.load(input)
  converted = value.to(_finitef.type.element_ty)
}

/-- error: finite helper and extern names cannot be rebound, aliased, or used outside direct calls -/
#guard_msgs in
example := pointerTypeCase%[tl.int1] {
  value = tl.load(input)
  converted = value.to(isfinite_func.type.element_ty)
}

#axiomsClean pointer_target_retains_explicit_conversion
#axiomsClean pointer_target_retains_fp64
#axiomsClean bool_pointer_matches_explicit
#axiomsClean bool_pointer_preserves_operand
end VeriTile.Tests.IsfinitePointerElementCast
