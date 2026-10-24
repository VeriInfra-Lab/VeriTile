import bench.repairs.tritonbench_g.rotary_emb.Kernels
import bench.repairs.tritonbench_g.rotary_emb.Proofs.TypedResult

namespace VeriTile.Bench.Repairs.RotaryEmb.KernelRealResult
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000

/-- Complete typed kernel correctness on arbitrary Real payloads. Both
interleaved columns use their own precomputed coefficients. The full memory
statement includes masked lanes, partial tails, and every untouched region. -/
theorem rotary_emb_real (rest : Algebra ℝ) (shape : Shape)
    (valid : shape.Valid) (s : State ℝ) :
    Spec.Real (∃ t,exec (Typed.algebra rest realOps) (Kernels.kernel (Typed.layout shape)) s=some t ∧
      t.mem=(Typed.target realOps shape s).mem ∧
      (∀ key odd i,Typed.active (Typed.layout shape) s key i →
        t.mem (Typed.region (Typed.layout shape) key) (Typed.offset (Typed.layout shape) s key odd i)=
          .mk .real (formula shape (Typed.inputs s) key
            (Typed.token (Typed.layout shape) s i) (Typed.head (Typed.layout shape) s i)
            (Typed.dimension (Typed.layout shape) odd i))) ∧
      (∀ r o,Typed.Outside shape s r o → t.mem r o=s.mem r o)) := by
  simpa only [Typed.value_formula] using
    Typed.successful rest realOps shape valid s


#specNonCircular formula avoiding [Kernels.kernel,Typed.execution,Typed.reference]
#axiomsClean rotary_emb_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmb.KernelRealResult
