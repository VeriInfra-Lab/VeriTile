import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Order

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.FPEquiv
open Checked Arena Binding
noncomputable section

/-- Checked canonical CUDA FP32 tensors; every raw FP32 word is admitted.
Q/K retain their distinct mul/neg/FMA operand orders, V is a raw-word copy.
The returned value is None; the original nine complete Tensor descriptors,
caller device, all written bytes, read-only inputs and exterior frame remain
part of the result. The source compiler/launcher boundary is conditional. -/
theorem fused_rotary_native (c : Call) (callerDevice : Nat) (p : Native.Instructions)
    (initial : Arena.Memory) (admitted : check c (mirror c initial)=true) :
    ∃ result,invoke c callerDevice p initial=some result ∧
      NativeResult c callerDevice p p initial result :=
  complete c callerDevice p initial admitted

/-- Only total local raw-word mul, neg, and FMA correspondences are deferred.
No exact-real, whole-output, metadata, addressing, or frame equation is assumed. -/
theorem fused_rotary_native_if_local (c : Call) (callerDevice : Nat)
    (actual target : Native.Instructions) (localAgreement : Native.Obligations actual target)
    (initial : Arena.Memory) (admitted : check c (mirror c initial)=true) :
    ∃ result,invoke c callerDevice actual initial=some result ∧
      NativeResult c callerDevice actual target initial result :=
  complete_if_local c callerDevice actual target localAgreement initial admitted

/-- The same complete finite-byte result for any legal complete work order. -/
theorem fused_rotary_native_any_order (c : Call) (callerDevice : Nat) (p : Native.Instructions)
    (initial : Arena.Memory) (admitted : check c (mirror c initial)=true)
    (order : List Work) (legal : (schedule c.shape).Perm order) :
    ∃ result,invokeOrder c callerDevice p order initial=some result ∧
      NativeResult c callerDevice p p initial result :=
  ordered_complete c callerDevice p initial admitted order legal

#axiomsClean fused_rotary_native_any_order
#axiomsClean fused_rotary_native
#axiomsClean fused_rotary_native_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.FPEquiv
