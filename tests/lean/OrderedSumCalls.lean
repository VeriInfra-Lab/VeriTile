import tests.lean.OrderedSumReference
import VeriTile.Triton.DSL.OrderedSumCalls
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.OrderedSumCalls
open VeriTile.Triton

def source (X Y : RegionName) (N B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call OrderedSumReference.signature call_args [N := N]
      ordered_sum_trees [total := tree] in triton {
    acc = tl.full([XBLOCK, B], 0, tl.float32)
    for base in range(0, N, B):
      lane = base + tl.arange(0, B)[None, :]
      value = tl.load(X + lane, mask=lane < N, other=0.0).to(tl.float32)
      acc = acc + value
    total = tl.sum(acc, 1)[:, None]
    tl.store(Y + tl.arange(0, 1)[:, None], total, None)
  }

/-- Every statement of the streamed program, including masks, cast, sum tree
and the final store, agrees for arbitrary dimensions and arbitrary tree. -/
theorem complete_program : source = OrderedSumReference.reference := by
  funext X Y N B tree
  rfl

/-- error: specialized natural constexpr arguments must remain immutable -/
#guard_msgs (error) in
example (X Y : RegionName) (N B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call OrderedSumReference.signature call_args [N := N]
      ordered_sum_trees [total := tree] in triton {
    XBLOCK = 2
    acc = tl.full([XBLOCK, B], 0, tl.float32)
    total = tl.sum(acc, 1)[:, None]
    tl.store(Y + tl.arange(0, 1)[:, None], total, None)
  }

/-- error: every source sum requires an exact ordered-tree binding -/
#guard_msgs (error) in
example (X Y : RegionName) (N B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call OrderedSumReference.signature call_args [N := N]
      ordered_sum_trees [total := tree] in triton {
    acc = tl.full([XBLOCK, B], 0, tl.float32)
    total = tl.sum(acc, 1)[:, None]
    extra = tl.sum(acc, 1)[:, None]
    tl.store(Y + tl.arange(0, 1)[:, None], total, None)
  }

/-- error: ordered sum requires the complete tl.sum(input, 1)[:, None] assignment -/
#guard_msgs (error) in
example (X Y : RegionName) (N B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call OrderedSumReference.signature call_args [N := N]
      ordered_sum_trees [total := tree] in triton {
    acc = tl.full([XBLOCK, B], 0, tl.float32)
    total = tl.sum(acc, 0)[:, None]
    tl.store(Y + tl.arange(0, 1)[:, None], total, None)
  }

/-- error: each ordered sum destination must occur exactly once -/
#guard_msgs (error) in
example (X Y : RegionName) (N B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call OrderedSumReference.signature call_args [N := N]
      ordered_sum_trees [total := tree] in triton {
    acc = tl.full([XBLOCK, B], 0, tl.float32)
    tl.store(Y, acc, None)
  }

#axiomsClean complete_program
#auditModuleAxioms
end VeriTile.Tests.OrderedSumCalls
