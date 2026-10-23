import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.FP
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Execution
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Tensor

namespace VeriTile.Bench.Repairs.KvCacheFilling.FPEquiv
open VeriTile.Triton

/-- Exact production precision: FP32 state, FP32 policy0 cache, uint8 policy4/8
cache, FP32 scale/zero; INT32 input metadata and INT64 schedule/pointer products.
The reference is concrete binary32 arithmetic and defined toward-zero uint8
conversion. Tree node, scalar primitive and uint8 packing laws remain explicit
local numerical obligations. They contain no memory or whole-output facts. -/
theorem whole_call_readback (r : Request) (d : Domain r) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations Word Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (m : Nat → Byte) (undefinedCast : Word → Nat)
    (numerics : TaskNumerics r ops (softwareOps undefinedCast) kp vp
      (readInput r inputs decodeWord (FP.Value.zero .fp32) (fun a => m a.row) Bool.false)
      (readInput r inputs decodeWord (FP.Value.zero .fp32) (fun a => m a.row) Bool.true))
    (_defined : TaskCastDomain r ops kp vp
      (readInput r inputs decodeWord (FP.Value.zero .fp32) (fun a => m a.row) Bool.false)
      (readInput r inputs decodeWord (FP.Value.zero .fp32) (fun a => m a.row) Bool.true))
    (j : Job) (hj : j ∈ jobs r) (a : Address) (ha : a ∈ rowAddresses r j)
    (b : Nat) (hb : b < elementBytes r a.role) :
    (call r l inputs ops kp vp kb vb wordByte decodeWord (FP.Value.zero .fp32) (fun a => m a.row)).1
      (physical (bytePtr r l a b)) =
      cellByte wordByte (taskValue r (softwareOps undefinedCast) kp vp
        (readInput r inputs decodeWord (FP.Value.zero .fp32) (fun a => m a.row) Bool.false)
        (readInput r inputs decodeWord (FP.Value.zero .fp32) (fun a => m a.row) Bool.true) j a) b := by
  exact (congrFun (call_execution r d l inputs ops kp vp kb vb wordByte decodeWord (FP.Value.zero .fp32) m)
    (bytePtr r l a b)).trans
    (whole_fp_bytes r d l ops kp vp _ _ m undefinedCast numerics j hj a ha b hb)

/-- Full caller backing-memory frame, including input states, metadata,
unwritten cache cells, offset prefixes/suffixes and unrelated live allocations.
This structural theorem has no floating-point numerical assumptions. -/
theorem whole_call_frame (r : Request) (d : Domain r) (l : Layout r) (inputs : InputLayout r l)
    (ops : Operations Word Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (m : Nat → Byte) (p : Nat)
    (outside : ∀ j ∈ jobs r, ∀ a ∈ rowAddresses r j, ∀ b, b < elementBytes r a.role → bytePtr r l a b ≠ p) :
    (call r l inputs ops kp vp kb vb wordByte decodeWord (FP.Value.zero .fp32) (fun a => m a.row)).1 (physical p) = m p := by
  exact (congrFun (call_execution r d l inputs ops kp vp kb vb wordByte decodeWord (FP.Value.zero .fp32) m) p).trans
    (whole_bytes_frame r l ops kp vp _ _ wordByte m p outside)

#axiomsClean whole_call_readback
#axiomsClean whole_call_frame

/-- Actual FP32/uint8 interpretation of all active rows, through caller Tensor
coordinates. Only local source-operand numerical conformance is conditional;
the complete memory frame and return behavior are unconditional. -/
theorem whole_tensor_fp32 (t : TensorInputs) (r : Request) (checked : checkInputs t r = Bool.true)
    (l : Layout r) (ins : InputLayout r l) (binding : Tensor.Bindings t r l ins)
    (ops : Operations Word Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (m : Nat → Byte) (objects : List Nat) (incoming : VeriTile.Torch.Device)
    (undefinedCast : Word → Nat)
    (numerics : TaskNumerics r ops (softwareOps undefinedCast) kp vp
      (Tensor.values decodeWord (FP.Value.zero .fp32) t r m Bool.false)
      (Tensor.values decodeWord (FP.Value.zero .fp32) t r m Bool.true))
    (_defined : TaskCastDomain r ops kp vp
      (Tensor.values decodeWord (FP.Value.zero .fp32) t r m Bool.false)
      (Tensor.values decodeWord (FP.Value.zero .fp32) t r m Bool.true)) :
    let result := Tensor.run r l ins ops kp vp kb vb wordByte decodeWord (FP.Value.zero .fp32) m objects incoming
    (∀ j∈jobs r, ∀ a∈rowAddresses r j, ∀ b,b<elementBytes r a.role →
      Tensor.byte (Tensor.output t a.role) result.memory (a.row*r.width a.role+a.column) b=
        cellByte wordByte (taskValue r (softwareOps undefinedCast) kp vp
          (Tensor.values decodeWord (FP.Value.zero .fp32) t r m Bool.false)
          (Tensor.values decodeWord (FP.Value.zero .fp32) t r m Bool.true) j a) b) ∧
    (∀ p,(∀ j∈jobs r, ∀ a∈rowAddresses r j, ∀ b,b<elementBytes r a.role →
      bytePtr r l a b≠p) → result.memory p=m p) ∧
    result.value=none ∧ result.objects=objects ∧ result.device=incoming := by
  have source := Tensor.whole_tensor_source t r checked l ins binding ops kp vp kb vb wordByte decodeWord
    (FP.Value.zero .fp32) m objects incoming
  refine ⟨?_,source.2⟩
  intro j hj a ha b hb
  rw [source.1 j hj a ha b hb,
    taskValue_fp r ops (softwareOps undefinedCast) kp vp _ _ j a ha (numerics j hj).1 (numerics j hj).2]

#axiomsClean whole_tensor_fp32
#specNonCircular taskValue avoiding [call,Tensor.run,runSourceJobs,executeBytes]
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.KvCacheFilling.FPEquiv
