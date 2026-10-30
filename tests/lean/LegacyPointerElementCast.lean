import VeriTile.Triton.DSL

namespace VeriTile.Tests.LegacyPointerElementCast
open VeriTile.Triton

/-! Historical mathematical surfaces have no declared storage profile.
Their pointer-element casts preserve the existing operand interpretation;
they do not provide fp64 provenance to the typed source classifier. -/
def legacy (A : RegionName) : ComputeKernel := triton {
  value = tl.load(A)
  copied = (value).to(A.type.element_ty)
}

def mathematical (A : RegionName) : ComputeKernel := triton {
  value = tl.load(A)
  copied = value
}

example (A : RegionName) : legacy A = mathematical A := rfl

def legacyBlock (A : RegionName) : ComputeKernel := triton {
  p = tl.make_block_ptr(base=A, shape=(4, 4), strides=(4, 1),
    offsets=(0, 0), block_shape=(1, 2), order=(1, 0))
  value = tl.load(p)
  copied = (value).to(p.type.element_ty)
}

def mathematicalBlock (A : RegionName) : ComputeKernel := triton {
  p = tl.make_block_ptr(base=A, shape=(4, 4), strides=(4, 1),
    offsets=(0, 0), block_shape=(1, 2), order=(1, 0))
  value = tl.load(p)
  copied = value
}

example (A : RegionName) : legacyBlock A = mathematicalBlock A := rfl

/-- error: pointer .type.element_ty cast requires known supported storage provenance -/
#guard_msgs in
def typedUnknown (A : RegionName) : ComputeKernel := triton {
  tl.region A = tl.float32
  value = tl.load(A)
  copied = (value).to(missing.type.element_ty)
}

def inlineStorage (R : RegionName) : ComputeKernel := triton {
  p = $((Region.cast R : Region .fp16)) + 0
  value = tl.load(p)
}

example (R : RegionName) :
    ∃ shape expr, (inlineStorage R).surfaceBody.getLast? =
      some (.assign .fp16 shape "value" expr) := ⟨_, _, rfl⟩

/-- error: pointer .type.element_ty cast requires known supported storage provenance -/
#guard_msgs in
def inlineRebound (R : RegionName) : ComputeKernel := triton {
  p = $((Region.cast R : Region .fp16)) + 0
  value = tl.load(p)
  p = 0
  copied = (value).to(missing.type.element_ty)
}

/-- error: pointer .type.element_ty cast requires known supported storage provenance -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.source_alias _finitef = triton.language.extra.cuda.libdevice.finitef
  tl.source_alias _isfinited = triton.language.extra.cuda.libdevice.isfinited
  tl.source_helper isfinite_func(x) {
    return (_isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32)))
  }
  value = (1.0).to(tl.float32)
  finite = isfinite_func(value)
  copied = (finite).to(missing.type.element_ty)
}

end VeriTile.Tests.LegacyPointerElementCast
