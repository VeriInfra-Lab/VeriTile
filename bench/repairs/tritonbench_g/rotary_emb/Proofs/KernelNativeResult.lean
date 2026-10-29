import bench.repairs.tritonbench_g.rotary_emb.Kernels
import bench.repairs.tritonbench_g.rotary_emb.Proofs.TypedNative
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Candidate

namespace VeriTile.Bench.Repairs.RotaryEmb.KernelNativeResult
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000

/-- Successful executions of the repair and its historical address-only
candidate have the same complete result. Floating outputs equal the independent
selected native graph on every 32-bit word. Primitive mul/neg/fma semantics
and coefficient loads form the model boundary; there are no deferred numerical
conformance hypotheses or finite-payload restrictions. -/
theorem rotary_emb_native_equiv (rest : Algebra Native.Node) (p : Native.Instructions)
    (shape : Shape) (valid : shape.Valid) (s : State Native.Node) :
    ∃ t,exec (Typed.algebra rest (Native.ops p)) (Kernels.kernel (Typed.layout shape)) s=some t ∧
      exec (Typed.algebra rest (Native.ops p)) (Candidate.optimized (Typed.layout shape)) s=some t ∧
      t.mem=(Typed.graphTarget p shape s).mem ∧
      (∀ key odd i,Typed.active (Typed.layout shape) s key i →
        t.mem (Typed.region (Typed.layout shape) key) (Typed.offset (Typed.layout shape) s key odd i)=
          Typed.graphValue p shape s key odd i) ∧
      (∀ r o,Typed.Outside shape s r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,source,memory,values,frame⟩ :=
    Typed.native_successful rest p shape valid s
  exact ⟨t,source,(Candidate.execution_equiv _ _ _).symm.trans source,memory,values,frame⟩


#specNonCircular Typed.graphTarget avoiding [Kernels.kernel,Candidate.optimized,Typed.execution]
#axiomsClean rotary_emb_native_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmb.KernelNativeResult
