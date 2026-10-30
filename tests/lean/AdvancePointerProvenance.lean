import VeriTile.Triton.DSL

namespace VeriTile.Tests.AdvancePointerProvenance
open VeriTile.Triton

def narrow (A : RegionName) (flag : Bool) : ComputeKernel := triton {
  tl.region A = tl.float16
  p = tl.make_block_ptr(base=A, shape=(4, 4), strides=(4, 1),
    offsets=(0, 0), block_shape=(1, 2), order=(1, 0))
  if flag {
    p = tl.advance(p, (0, 2))
  }
  value = tl.load(p, boundary_check=(0, 1), padding_option="zero")
}

example (A : RegionName) (flag : Bool) :
    ∃ shape expr, (narrow A flag).surfaceBody.getLast? =
      some (.assign .fp16 shape "value" expr) := ⟨_, _, rfl⟩

def wide (A : RegionName) (flag : Bool) : ComputeKernel := triton {
  tl.region A = tl.float64
  p = tl.make_block_ptr(base=A, shape=(4, 4), strides=(4, 1),
    offsets=(0, 0), block_shape=(1, 2), order=(1, 0))
  if flag {
    p = tl.advance(p, [0, 2])
  } else {
    p = tl.advance(p, offsets=(0, 1))
  }
  value = tl.load(p, boundary_check=(0, 1), padding_option="zero")
  finite = tl.extra.cuda.libdevice.isfinited(value)
}

example (A : RegionName) (flag : Bool) :
    ∃ shape expr, (wide A flag).surfaceBody.getLast? =
      some (.assign .bool shape "finite" (.isFinite .fp64 expr)) := ⟨_, _, rfl⟩

/-- error: pointer storage dtype conflict across branches: p -/
#guard_msgs in
def conflict (A B : RegionName) (flag : Bool) : ComputeKernel := triton {
  tl.region A = tl.float16, B = tl.bfloat16
  a = tl.make_block_ptr(base=A, shape=(4, 4), strides=(4, 1),
    offsets=(0, 0), block_shape=(1, 2), order=(1, 0))
  b = tl.make_block_ptr(base=B, shape=(4, 4), strides=(4, 1),
    offsets=(0, 0), block_shape=(1, 2), order=(1, 0))
  if flag { p = tl.advance(a, (0, 2)) } else { p = tl.advance(b, (0, 2)) }
  value = tl.load(p)
}

end VeriTile.Tests.AdvancePointerProvenance
