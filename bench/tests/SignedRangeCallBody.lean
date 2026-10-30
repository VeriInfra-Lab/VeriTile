import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedRangeCallBody
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("out", .fp32)]
  scalars := [("n", .i64)]
  constexpr := ["BLOCK"]
  writePorts := ["out"]

/-- Preserve the signed constructor statements. Only their provably immutable
forward stride is propagated into the existing signed-range instruction. -/
def forward (out : RegionName) (n : Nat) : ComputeKernel :=
  let BLOCK : Nat := 4
  with_triton_call signature call_args [n := n] in
  triton {
    N = tl.cast(n, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), BLOCK, tl.int64)
    for i in range(zero, N, step):
        tl.store(out + i, 1.0)
  }

theorem signed_loop (out : RegionName) (n : Nat) :
    (match ((forward out n).surfaceBody : List ComputeStmt)[3]? with
    | some (ComputeStmt.forRangeInt "i" (.ref .int [] "zero") (.ref .int [] "N")
        (.constNat 4) _body) => Bool.true
    | _ => Bool.false) = Bool.true := rfl

theorem kept_step_assignment (out : RegionName) (n : Nat) :
    (forward out n).surfaceBody[2]? = some
      (.assign .int [] "step" (.alg (.full [] (.constInt 4)))) := rfl

-- A known rebinding before loop entry changes the captured stride. Preserve
-- both assignments and prove the entire program equals its literal-step form.
def rebound (out : RegionName) (n : Nat) : ComputeKernel :=
  let BLOCK : Nat := 4
  with_triton_call signature call_args [n := n] in
  triton {
    N = tl.cast(n, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), BLOCK, tl.int64)
    step = -1
    for i in range(zero, N, step):
        tl.store(out + i, 1.0)
  }

def reboundLiteral (out : RegionName) (n : Nat) : ComputeKernel :=
  let BLOCK : Nat := 4
  with_triton_call signature call_args [n := n] in
  triton {
    N = tl.cast(n, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), BLOCK, tl.int64)
    step = -1
    for i in range(zero, N, -1):
        tl.store(out + i, 1.0)
  }

theorem known_rebinding (out : RegionName) (n : Nat) :
    rebound out n = reboundLiteral out n := rfl
#axiomsClean known_rebinding

-- A loop-carried unknown stride is still rejected.
/-- error: nat expression: dtype mismatch -/
#guard_msgs (error) in
example (out : RegionName) (n : Nat) : ComputeKernel :=
  let BLOCK : Nat := 4
  with_triton_call signature call_args [n := n] in
  triton {
    N = tl.cast(n, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), BLOCK, tl.int64)
    for i in range(zero, N, step):
        step, unused = -1, 0
        tl.store(out + i, 1.0)
  }

-- Unknown symbolic Nat is not a checked block size.
/-- error: nat expression: dtype mismatch -/
#guard_msgs (error) in
example (out : RegionName) (n BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in
  triton {
    N = tl.cast(n, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), BLOCK, tl.int64)
    for i in range(zero, N, step):
        tl.store(out + i, 1.0)
  }

/-- N is read but never assigned by the loop. Its explicit signed-cast dtype
survives both zero and nonzero iteration counts, so floating division remains
floating in the following loop. -/
def immutable_count (out : RegionName) (n : Nat) : ComputeKernel :=
  let BLOCK : Nat := 4
  with_triton_call signature call_args [n := n] in
  triton {
    N = tl.cast(n, tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, n):
        acc += 1.0
    mean = tl.sum(acc, axis=0) / N
    for i in range(0, n):
        acc += mean
    tl.store(out + tl.arange(0, BLOCK), acc)
  }

/-- error: loop-carried value requires a stable, known source dtype -/
#guard_msgs (error) in
example (out : RegionName) (n : Nat) : ComputeKernel :=
  let BLOCK : Nat := 4
  with_triton_call signature call_args [n := n] in
  triton {
    N = tl.cast(n, tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, n):
        N = 1.0
        acc += 1.0
    mean = tl.sum(acc, axis=0) / N
    for i in range(0, n):
        acc += mean
    tl.store(out + tl.arange(0, BLOCK), acc)
  }

/-- A signed divisor must not make the later explicitly FP32 variance
accumulator an integer. This was a whole-body dtype propagation regression. -/
def floating_variance (out : RegionName) (n : Nat) : ComputeKernel :=
  let BLOCK : Nat := 4
  with_triton_call signature call_args [n := n] in
  triton {
    N = tl.cast(n, tl.int64)
    _mean = tl.zeros([BLOCK], dtype=tl.float32)
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, n):
        x = tl.full([BLOCK], 1.0, tl.float32)
        x = tl.where(tl.arange(0, BLOCK) < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    tl.store(out, var)
  }

theorem variance_dtype (out : RegionName) (n : Nat) :
    (match ((floating_variance out n).surfaceBody : List ComputeStmt)[3]? with
    | some (.assign .real [4] "_var" _) => Bool.true
    | _ => Bool.false) = Bool.true := rfl

#axiomsClean variance_dtype
#axiomsClean signed_loop
#axiomsClean kept_step_assignment
end
end VeriTile.Tests.SignedRangeCallBody
