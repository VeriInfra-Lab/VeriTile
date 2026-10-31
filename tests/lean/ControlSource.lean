import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.ControlSource
import bench.tritonbench_g.spinning_lock_reduction.SourceAdapter
import bench.tritonbench_g.spinning_lock_reduction.SourceSyntax
import bench.tritonbench_g.streamk_matmul.SourceSyntax
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ControlSource
open VeriTile.Triton.ControlSource
open VeriTile.Bench.Repairs.SpinningLock

def helperCalls : Program := source_program triton {
  for i in range(0, M) { helper(A, B, i) }
  pid_m, pid_n = mapping(tile_id, M, N)
  return pid_m, pid_n
}

private def outline : Statement → String
  | .range _ _ _ => "range"
  | .bindCall _ _ _ => "bind"
  | .returnValues _ => "return"
  | _ => "other"

theorem helper_control_retained : helperCalls.map outline = ["range", "bind", "return"] := rfl

private def argumentCount : Statement → List Nat
  | .call _ args | .bindCall _ _ args => [args.length]
  | .returnValues values => [values.length]
  | _ => []

private def helperArgumentCounts : Statement → List Nat
  | .range _ bounds body => bounds.length :: body.flatMap argumentCount
  | other => argumentCount other

theorem complete_helper_arguments : helperCalls.flatMap helperArgumentCounts = [2,3,3,2] := rfl

def atomicSource : Program := source_program triton {
  while tl.atomic_cas(locks + next_pid, 1, 7) != 1 { pass }
  tl.debug_barrier()
}

/-- The replacement and expected old word stay distinct, and the barrier
remains a separate event after the spin loop. -/
theorem complete_cas_operands : SourceAdapter.decode atomicSource = .ok [
    .whileLoop (.casNe (.offset (.base .locks) (.variable "next_pid"))
      (.literal 1) (.literal 7) (.literal 1)) [.pass],
    .blockBarrier] := rfl

private def rejected (result : Except String α) : Bool :=
  match result with | .error _ => Bool.true | .ok _ => Bool.false

def wrongPrecision : Program := source_program triton {
  acc = tl.zeros([BM, BN], dtype=tl.float16)
}

theorem changed_precision_rejected : rejected (SourceAdapter.decode wrongPrecision) = Bool.true := rfl

/-- Importing the control frontend must also keep ordinary masked source
bodies parseable through the shared load/store keyword grammar. -/
def regularMaskedSource (p : VeriTile.Triton.RegionName) : VeriTile.Triton.ComputeKernel := triton {
  r = tl.arange(0, $(4))
  x = tl.load(p + r, mask=r < $(3), other=0.0)
  tl.store(p + r, x, mask=r < $(3))
}

/-- Importing both existing frontends and the shared frontend must preserve
ordinary Lean Boolean terms without duplicate notation alternatives. -/
theorem joint_boolean_terms : (true, false) = (Bool.true, Bool.false) := rfl

#axiomsClean helper_control_retained
#axiomsClean complete_helper_arguments
#axiomsClean complete_cas_operands
#axiomsClean changed_precision_rejected
#axiomsClean joint_boolean_terms
end VeriTile.Tests.ControlSource
