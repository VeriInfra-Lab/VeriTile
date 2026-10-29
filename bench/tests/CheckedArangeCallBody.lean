import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.CheckedArangeCallBody
open VeriTile.Triton FP.Structural
noncomputable section

def signature : CallSignature where
  pointers := [("out",.fp32)]
  scalars := [("seed",.i64)]
  constexpr := ["BLOCK"]

/-- The checked bound is supplied by the caller. The body is plain Triton. -/
def sample (out : RegionName) (seed : Int) (block : Nat)
    (valid : Random.ArangeSizeDomain block) : ComputeKernel :=
  let checked := Random.i32ArangeSize block valid
  let forwarded := checked
  let BLOCK := forwarded
  with_triton_call signature call_args [seed := seed] in
  triton {
    cols = tl.arange(0, BLOCK)
    value = tl.rand(seed, cols, n_rounds=7)
    tl.store(out + cols, value)
  }

/-- Alias substitution preserves both the actual symbolic bound and the checked
signed32 offset profile in the RNG instruction. It inserts no body statement. -/
theorem symbolic_rng_profile (out : RegionName) (seed : Int) (block : Nat)
    (valid : Random.ArangeSizeDomain block) :
    (sample out seed block valid).surfaceBody[1]? = some (
      .assign .real [block] "value" (.compute (.alg .fp32
        (.rand { seedType := .i64, offsetType := .i32, nRounds := 7 }
          (.constInt seed) (.castNatToInt (.ref .nat [block] "cols")))))) := rfl

-- Merely having a bound proof in scope cannot label an unadapted Nat.
/-- error: tl.rand: offsets have no supported source integer profile; unknown symbolic sizes need a checked rooted i32ArangeSize adapter -/
#guard_msgs (error) in
example (out : RegionName) (seed : Int) (BLOCK : Nat)
    (_valid : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call signature call_args [seed := seed] in
  triton {
    cols = tl.arange(0, BLOCK)
    value = tl.rand(seed, cols, n_rounds=7)
    tl.store(out + cols, value)
  }

-- A local function with the adapter's short name is not the checked constant.
/-- error: tl.rand: offsets have no supported source integer profile; unknown symbolic sizes need a checked rooted i32ArangeSize adapter -/
#guard_msgs (error) in
example (out : RegionName) (seed : Int) (block : Nat)
    (i32ArangeSize : Nat → Nat) : ComputeKernel :=
  let BLOCK := i32ArangeSize block
  with_triton_call signature call_args [seed := seed] in
  triton {
    cols = tl.arange(0, BLOCK)
    value = tl.rand(seed, cols, n_rounds=7)
    tl.store(out + cols, value)
  }

#axiomsClean symbolic_rng_profile
end
end VeriTile.Tests.CheckedArangeCallBody
