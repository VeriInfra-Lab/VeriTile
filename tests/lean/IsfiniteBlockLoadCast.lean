import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! Uncompiled focused regressions for the direct-load pointer-element cast.
The actual checked helper consumes each loaded result, so losing precision
provenance cannot silently pass as an unused AST fragment. -/
namespace VeriTile.Tests.IsfiniteBlockLoadCast
open Lean VeriTile.Triton

set_option hygiene false in
macro "finiteBlockLoad%" "[" storage:tritonDType "]" mode:ident : term => do
  let loaded ←
    if mode.getId == `attr then
      `(tritonExpr| tl.load(p, boundary_check=([0] : List Nat)).to(input.type.element_ty))
    else
      `(tritonExpr| tl.load(p, boundary_check=([0] : List Nat)).to($storage:tritonDType))
  `(fun (input : RegionName) => triton {
    tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
    tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
    tl.source_helper isfinite_func(x) {
      return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
    }
    tl.region input = $storage
    p = tl.make_block_ptr(base=input, shape=($(3)), strides=($(1)),
      offsets=($(0)), block_shape=($(4)), order=(0))
    value = $loaded:tritonExpr
    finite = isfinite_func(value)
  })

theorem fp32_load_cast_exact :
    finiteBlockLoad%[tl.float32] attr = finiteBlockLoad%[tl.float32] explicit := rfl

theorem fp64_load_cast_exact :
    finiteBlockLoad%[tl.float64] attr = finiteBlockLoad%[tl.float64] explicit := rfl

theorem bf16_load_cast_exact :
    finiteBlockLoad%[tl.bfloat16] attr = finiteBlockLoad%[tl.bfloat16] explicit := rfl

#axiomsClean fp32_load_cast_exact
#axiomsClean fp64_load_cast_exact
#axiomsClean bf16_load_cast_exact
end VeriTile.Tests.IsfiniteBlockLoadCast
