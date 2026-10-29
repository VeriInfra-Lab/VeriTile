import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Ideal
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.DirectReal

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section

/-- Full wrapper under its ideal Real interpretation. The target was defined
independently of source schedules, tile sizes, compiler plans and split count.
FP/native equality to this Real expression is deliberately not claimed. -/
theorem correct (s : Shape) (b : Buffers) (fresh : b.Fresh) (hasBias : Bool)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s)
    (initial : Memory ℝ ℝ) (x : MatrixCell s) :
    let returned := wrapper realOps s b (1/16129) hasBias owner callerDevice benchmarks chosen initial
    returned.tensor=outputTensor s b owner ∧ returned.currentDevice=callerDevice ∧
    read16 realOps returned.memory (outputAddress s b x)=target s (idealInput s b hasBias initial) x.1 x.2 ∧
    InputsAgree b returned.memory initial := by
  dsimp only
  obtain ⟨ret,dev⟩ := wrapper_returns realOps s b (1/16129) hasBias owner callerDevice benchmarks chosen initial
  refine ⟨ret,dev,?_,wrapper_frame _ _ _ _ _ _ _ _ _ _⟩
  rw [wrapper_output _ _ _ fresh,real_reference]

namespace Specification
open Direct

theorem python_checks_iff_preconditions (a b x w : Argument) (bias : BiasArgument) :
    checkInputs a b x w bias=.ok () ↔ AdmittedArguments a b x w bias := checkInputs_iff a b x w bias

/-- Complete checked wrapper, original tensor values, arbitrary real scales
and bias, exact signed INT8 contraction, device restoration and caller frame. -/
theorem int8_dequant_matmul_wrapper_correct (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (callerDevice : Nat) (benchmarks : List (Invocation (requestShape r))) (chosen : Invocation (requestShape r))
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (inputs : InputsLive r live) (alloc : Allocation r caller live) :
    ∃ result, run r realOps (1/16129) callerDevice benchmarks chosen caller live alloc=.ok result ∧
      result.tensor=outputTensor (requestShape r) (requestBuffers r alloc.storage) r.a.info.metadata.device.index ∧
      result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell (requestShape r),read16 realOps result.memory (outputAddress (requestShape r) (requestBuffers r alloc.storage) cell)=
        target (requestShape r) (logicalInput r caller) cell.1 cell.2) ∧
      (∀ p,live p.1 → result.memory p=caller p) :=
  real_complete base r laws callerDevice benchmarks chosen caller live inputs alloc

#axiomsClean python_checks_iff_preconditions
#axiomsClean int8_dequant_matmul_wrapper_correct
end Specification

#axiomsClean correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
