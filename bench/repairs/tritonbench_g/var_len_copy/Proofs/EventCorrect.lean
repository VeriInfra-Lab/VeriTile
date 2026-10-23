import bench.repairs.tritonbench_g.var_len_copy.Proofs.Real
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.VarLenCopy.Correct
noncomputable section

/-- Exact dynamic/typed Python admission; non-Tensors have no metadata. -/
theorem event_python_checks_iff (m : Option Inputs) (segments : Nat → Segment) :
    checkArguments m segments = .ok () ↔ AdmittedArguments m segments :=
  checkArguments_iff m segments

/-- Arbitrary Real payloads with the actual signed-int32 source metadata trace.
The contract is exactly the guarded direct contiguous-buffer wrapper contract;
no interpretation of Real through finite FP32 words is required. -/
specification event_var_len_copy_real (m : Inputs) (segments : Nat → Segment)
    (h : InputContract m segments) (metadata : Memory)
    (binding : TensorBinding m segments metadata) (initial : Nat → ℝ) :
    SegmentCopy (request m segments)
      (fun i => initial (addr (base m .source) i))
      (fun i => initial (addr (base m .output) i))
      (fun i => Mathematical.run (directLayout m) metadata initial (addr (base m .output) i)) ∧
    (∀ cell, (¬∃ x, Touched (request m segments) x ∧ cell=addr (base m .output) x) →
      Mathematical.run (directLayout m) metadata initial cell=initial cell) :=
  Mathematical.correct (directLayout m) (request m segments) metadata
    (prepared_direct m segments h metadata binding) (contract_valid m segments h) initial

#axiomsClean event_python_checks_iff
#axiomsClean event_var_len_copy_real
#specNonCircular SegmentCopy avoiding [Mathematical.run,launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy.Correct
