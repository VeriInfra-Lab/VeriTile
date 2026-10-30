import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PointerElementCast
open Lean VeriTile.Triton

set_option hygiene false in
macro "pointerCast%" "[" dt:tritonDType "]" version:ident : term => do
  let value : TSyntax `tritonExpr ←
    if version.getId == `attr then `(tritonExpr| (value).to(p.dtype.element_ty))
    else `(tritonExpr| (value).to($dt:tritonDType))
  `(fun (dtype out_dtype : FloatDType) (X : RegionName) => triton {
    tl.region X = $dt
    p = tl.make_block_ptr(base=X, shape=($(4)), strides=($(1)),
      offsets=($(0)), block_shape=($(4)), order=(0))
    value = tl.full([$(4)], 1.0, dtype=tl.float32)
    tl.store(p, $value, boundary_check=([0] : List Nat))
  })

/-- Every concrete storage type and both dynamic dtype binders resolve at the
block pointer, retaining the same conversion as an explicit destination type. -/
theorem all_storage_casts :
    pointerCast%[tl.float64] attr = pointerCast%[tl.float64] explicit ∧
    pointerCast%[tl.float32] attr = pointerCast%[tl.float32] explicit ∧
    pointerCast%[tl.float16] attr = pointerCast%[tl.float16] explicit ∧
    pointerCast%[tl.bfloat16] attr = pointerCast%[tl.bfloat16] explicit ∧
    pointerCast%[tl.float8e4nv] attr = pointerCast%[tl.float8e4nv] explicit ∧
    pointerCast%[tl.float8e5] attr = pointerCast%[tl.float8e5] explicit ∧
    pointerCast%[DTYPE] attr = pointerCast%[DTYPE] explicit ∧
    pointerCast%[OUT_DTYPE] attr = pointerCast%[OUT_DTYPE] explicit :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

set_option hygiene false in
macro "rebound%" version:ident : term => do
  let first : TSyntax `tritonExpr ←
    if version.getId == `attr then `(tritonExpr| (value).to(p.dtype.element_ty))
    else `(tritonExpr| (value).to(tl.float16))
  let second : TSyntax `tritonExpr ←
    if version.getId == `attr then `(tritonExpr| (value).to(p.dtype.element_ty))
    else `(tritonExpr| (value).to(tl.float64))
  `(fun (A B : RegionName) => triton {
    tl.region A = tl.float16, B = tl.float64
    p = A + $(0)
    value = tl.full([], 1.0, dtype=tl.float32)
    x = $first
    p = B + $(0)
    y = $second
  })

theorem uses_current_binding : rebound% attr = rebound% explicit := rfl

#axiomsClean all_storage_casts
#axiomsClean uses_current_binding
end VeriTile.Tests.PointerElementCast
