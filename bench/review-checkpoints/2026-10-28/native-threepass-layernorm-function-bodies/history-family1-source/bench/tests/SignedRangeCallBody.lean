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

-- A second assignment invalidates constant propagation, even with the same
-- initial full constructor. Unknown signed strides remain rejected.
/-- error: nat expression: dtype mismatch -/
#guard_msgs (error) in
example (out : RegionName) (n : Nat) : ComputeKernel :=
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

#axiomsClean signed_loop
#axiomsClean kept_step_assignment
end
end VeriTile.Tests.SignedRangeCallBody
