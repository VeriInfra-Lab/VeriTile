import tests.lean.StaticRangesReference
import VeriTile.Triton.DSL.StaticRanges
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.StaticRanges
open VeriTile.Triton StaticRangesReference

def copied (Out : RegionName) (HEAD_DIM KCACHE_X : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
        offsets = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
        tl.store(Out + offsets, offsets)
  }

/-- An independently elaborated complete typed program checks both the
constexpr trip count and every nonzero partition's lane/store address. -/
theorem whole_typed_program (Out : RegionName) (HEAD_DIM KCACHE_X : Nat) :
    copied Out HEAD_DIM KCACHE_X=reference Out HEAD_DIM KCACHE_X := by rfl

example (Out : RegionName) (HEAD_DIM KCACHE_X dynamic : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call signature call_args [] in triton {
      for split_x in tl.static_range(dynamic // KCACHE_X):
          offsets = tl.arange(0, KCACHE_X)
          tl.store(Out + offsets, offsets)
    }
  trivial

example (Out : RegionName) (HEAD_DIM KCACHE_X : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call signature call_args [] in triton {
      HEAD_DIM = KCACHE_X
      for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
          offsets = tl.arange(0, KCACHE_X)
          tl.store(Out + offsets, offsets)
    }
  trivial

example (Out : RegionName) (HEAD_DIM KCACHE_X : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call signature call_args [] in triton {
      for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
          split_x = -1
          offsets = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
          tl.store(Out + offsets, offsets)
    }
  trivial

#axiomsClean whole_typed_program
end VeriTile.Tests.StaticRanges
