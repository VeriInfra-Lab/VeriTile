import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Real
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Execution
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Tensor
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Ideal

namespace VeriTile.Bench.Repairs.KvCacheFilling.Correct

/-- Independent affine quantization target: extrema over the complete real
head; (max-min)/levels, constant scale 1, zero -min/scale, nearest nonnegative
integer code, and low-half + 16*high-half uint8 packing. FP32 accuracy is a
separate question. This is the Real interpretation of the shared source. -/
noncomputable def target (r : Request) (k v : Nat → ℝ) (kl kh vl vh : ℝ) (a : Address) : Cell ℝ Byte :=
  rowValue (idealRow r.policy r.ck k kl kh) (idealRow r.policy r.cv v vl vh) a

/-- All symbolic dimensions and every validated token/head, with both cache
and corresponding scale/zero output roles. Descriptors include actual storage
offsets and complete allocation capacities; source bindings fix the ordered
compiler reduction trees and masks. The encode/decode ports are the Real
projection's value transport, not an assertion that arbitrary reals are FP32. -/
theorem whole_call_readback (r : Request) (d : Domain r) (l : Layout r) (inputs : InputLayout r l)
    (kp : Plan r.dk) (vp : Plan r.dv) (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : ℝ → Nat → Byte) (decode : (Fin 4 → Byte) → ℝ) (m : Nat → Byte)
    (j : Job) (hj : j ∈ jobs r) (a : Address) (ha : a ∈ rowAddresses r j)
    (b : Nat) (hb : b < elementBytes r a.role) (kl kh vl vh : ℝ)
    (hk : Extrema r.dk (readInput r inputs decode 0 (fun a => m a.row) Bool.false (sourceRow r j)) kl kh)
    (hv : Extrema r.dv (readInput r inputs decode 0 (fun a => m a.row) Bool.true (sourceRow r j)) vl vh) :
    (call r l inputs realOps kp vp kb vb encode decode 0 (fun a => m a.row)).1 (physical (bytePtr r l a b)) =
      cellByte encode (target r
        (readInput r inputs decode 0 (fun a => m a.row) Bool.false (sourceRow r j))
        (readInput r inputs decode 0 (fun a => m a.row) Bool.true (sourceRow r j)) kl kh vl vh a) b := by
  have h := congrFun (call_execution r d l inputs realOps kp vp kb vb encode decode 0 m) (bytePtr r l a b)
  change (call r l inputs realOps kp vp kb vb encode decode 0 (fun a => m a.row)).1 (physical (bytePtr r l a b)) = _ at h
  rw [h,whole_bytes_readback r d l realOps kp vp _ _ encode m j hj a ha b hb]
  simp only [taskValue,target,rowKernel_real kp _ _ _ kl kh hk,rowKernel_real vp _ _ _ vl vh hv]

theorem whole_call_frame (r : Request) (d : Domain r) (l : Layout r) (inputs : InputLayout r l)
    (kp : Plan r.dk) (vp : Plan r.dv) (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : ℝ → Nat → Byte) (decode : (Fin 4 → Byte) → ℝ) (m : Nat → Byte) (p : Nat)
    (outside : ∀ j ∈ jobs r, ∀ a ∈ rowAddresses r j, ∀ b, b < elementBytes r a.role → bytePtr r l a b ≠ p) :
    (call r l inputs realOps kp vp kb vb encode decode 0 (fun a => m a.row)).1 (physical p) = m p := by
  exact (congrFun (call_execution r d l inputs realOps kp vp kb vb encode decode 0 m) p).trans
    (whole_bytes_frame r l realOps kp vp _ _ encode m p outside)

#axiomsClean whole_call_readback
#axiomsClean whole_call_frame

/-- Guarded Real interpretation of the repaired call, with observations through
caller Tensor pointers, complete frame, None, unchanged object identities and
restored CUDA device context. Real arithmetic is a separate interpretation. -/
theorem whole_tensor_real (t : TensorInputs) (r : Request) (checked : checkInputs t r = Bool.true)
    (l : Layout r) (ins : InputLayout r l) (binding : Tensor.Bindings t r l ins)
    (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : ℝ → Nat → Byte) (decode : (Fin 4 → Byte) → ℝ) (m : Nat → Byte)
    (objects : List Nat) (incoming : VeriTile.Torch.Device)
    (kl kh vl vh : Job → ℝ)
    (hk : ∀ j∈jobs r, Extrema r.dk (Tensor.values decode 0 t r m Bool.false (sourceRow r j)) (kl j) (kh j))
    (hv : ∀ j∈jobs r, Extrema r.dv (Tensor.values decode 0 t r m Bool.true (sourceRow r j)) (vl j) (vh j)) :
    let result := Tensor.run r l ins realOps kp vp kb vb encode decode 0 m objects incoming
    (∀ j∈jobs r, ∀ a∈rowAddresses r j, ∀ b,b<elementBytes r a.role →
      Tensor.byte (Tensor.output t a.role) result.memory (a.row*r.width a.role+a.column) b=
        cellByte encode (target r (Tensor.values decode 0 t r m Bool.false (sourceRow r j))
          (Tensor.values decode 0 t r m Bool.true (sourceRow r j)) (kl j) (kh j) (vl j) (vh j) a) b) ∧
    (∀ p,(∀ j∈jobs r, ∀ a∈rowAddresses r j, ∀ b,b<elementBytes r a.role →
      bytePtr r l a b≠p) → result.memory p=m p) ∧
    result.value=none ∧ result.objects=objects ∧ result.device=incoming := by
  have source := Tensor.whole_tensor_source t r checked l ins binding realOps kp vp kb vb encode decode 0 m objects incoming
  refine ⟨?_,source.2⟩
  intro j hj a ha b hb
  rw [source.1 j hj a ha b hb]
  simp only [taskValue,target,rowKernel_real kp _ _ _ (kl j) (kh j) (hk j hj),
    rowKernel_real vp _ _ _ (vl j) (vh j) (hv j hj)]

#axiomsClean whole_tensor_real

/-- Whole-call mathematical Tensor result over arbitrary Real inputs. This
result observes actual logical Real cache/scale values and exact quantized
bytes directly; it has no encoder, decoder or finite-representation premise. -/
theorem whole_mathematical_real (t : TensorInputs) (r : Request)
    (checked : checkInputs t r=Bool.true) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (s : Ideal.State) (objects : List Nat) (incoming : VeriTile.Torch.Device)
    (kl kh vl vh : Job → ℝ)
    (hk : ∀ j∈jobs r, Extrema r.dk (s.key (sourceRow r j)) (kl j) (kh j))
    (hv : ∀ j∈jobs r, Extrema r.dv (s.value (sourceRow r j)) (vl j) (vh j)) :
    let result := Ideal.call r kp vp kb vb s objects incoming
    (∀ j∈jobs r, ∀ a∈rowAddresses r j,
      result.state.cache a=target r (s.key (sourceRow r j)) (s.value (sourceRow r j))
        (kl j) (kh j) (vl j) (vh j) a) ∧
    (∀ a,(∀ j∈jobs r,a∉rowAddresses r j) → result.state.cache a=s.cache a) ∧
    result.state.key=s.key ∧ result.state.value=s.value ∧
    result.returned=none ∧ result.objects=objects ∧ result.device=incoming := by
  have h := (checkInputs_iff t r).mp checked
  have exec := Ideal.call_execution r kp vp kb vb s objects incoming
  refine ⟨?_,?_,exec.1,exec.2.1,rfl,rfl,rfl⟩
  · intro j hj a ha
    rw [exec.2.2,whole_readback r (Native.domain h) realOps kp vp s.key s.value s.cache j hj a ha]
    simp only [taskValue,target,rowKernel_real kp _ _ _ (kl j) (kh j) (hk j hj),
      rowKernel_real vp _ _ _ (vl j) (vh j) (hv j hj)]
  · intro a outside
    rw [exec.2.2]
    exact whole_frame r realOps kp vp s.key s.value s.cache a outside

#axiomsClean whole_mathematical_real
#specNonCircular target avoiding [Ideal.call,Ideal.runJobs,call,execute]
#specNonCircular target avoiding [call,Tensor.run,runSourceJobs,executeBytes]
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.KvCacheFilling.Correct
