import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.WholeBacking
import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.Proofs.RawCorrect
import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.Proofs.RawFPEquiv
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Real
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Execution
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Tensor
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Ideal
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.FP

/-! Optional historical whole-call and Tensor application results. Their original
statements, numerical premises and proofs are retained separately from the
selected shared-kernel certificates. These are not new numerical deferrals. -/

namespace VeriTile.Bench.Repairs.KvCacheFilling.TensorBridge.Real

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

end VeriTile.Bench.Repairs.KvCacheFilling.TensorBridge.Real
namespace VeriTile.Bench.Repairs.KvCacheFilling.TensorBridge.Native
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

end VeriTile.Bench.Repairs.KvCacheFilling.TensorBridge.Native
/-! Full current Tensor application of the raw native kernel certificate. -/
namespace VeriTile.Bench.Repairs.KvCacheFilling.TensorBridge.Native
open VeriTile.Triton FP.Structural
noncomputable section

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def nativeTarget (a : NativeProfile.Arithmetic) (reductions : QuantNative.Reductions)
    (t : TensorInputs) (r : Request) (m : Nat → Byte) (j : Job) (address : Address) (b : Nat) : Byte :=
  cellByte wordByte (SharedTarget.value (QuantNative.ops a reductions) r
    (SourceTensor.row decodeWord t r m j Bool.true)
    (SourceTensor.row decodeWord t r m j Bool.false) address) b

/-- Complete shared source/candidate execution and one actual caller heap
for each version. Native primitives preserve FP32 special values and complete
ordered reductions; the emitted uint8 conversion is the total PTX model.
The result has no numerical-agreement, finite-payload, defined-cast, output,
or successful-execution premise. `schedule.metadata.strides` contains the
actual Tensor.stride query observations paired with the queried descriptors;
the shared TensorMetadata type omits these fields, so this input ABI supplies
them explicitly. All metadata and scheduler values are decoded from `m`.
Host allocation and launch realization are outside this kernel theorem. -/
theorem whole_shared_native (a : NativeProfile.Arithmetic) (reductions : QuantNative.Reductions)
    (t : TensorInputs) (r : Request) (checked : checkInputs t r = Bool.true)
    (l : Layout r) (ins : InputLayout r l) (binding : Tensor.Bindings t r l ins)
    (m : Nat → Byte) (schedule : SourceTensor.Schedule t r m) (scratch : State Word) :
    ∃ states : Bool → Job → State Word, ∃ post : Bool → Nat → Byte,
      (∀ version j, j ∈ jobs r →
        exec (QuantProfile.algebra (QuantNative.ops a reductions)) (Shared.kernel version r)
          (SourceTensor.initial decodeWord t r m schedule scratch j) = some (states version j)) ∧
      (∀ version j, j ∈ jobs r → ∀ reg i,
        SharedReadback.Outside r (destinationRow r j) reg i →
          (states version j).mem reg i =
            (SourceTensor.initial decodeWord t r m schedule scratch j).mem reg i) ∧
      (∀ version, post version = SourceBacking.memory r l (states version) wordByte m) ∧
      (∀ version j, j ∈ jobs r → ∀ address, address ∈ rowAddresses r j → ∀ b,
        b < elementBytes r address.role →
          Tensor.byte (Tensor.output t address.role) (post version)
            (address.row*r.width address.role+address.column) b =
              nativeTarget a reductions t r m j address b) ∧
      (∀ version p,
        (∀ j ∈ jobs r, ∀ address ∈ rowAddresses r j, ∀ b,
          b < elementBytes r address.role → bytePtr r l address b ≠ p) → post version p = m p) ∧
      post Bool.false = post Bool.true := by
  have h : InputContract t r := (checkInputs_iff t r).mp checked
  let initial := fun j => SourceTensor.initial decodeWord t r m schedule scratch j
  obtain ⟨states,ran,frame,values,stores⟩ := RawNative.kernel_native a reductions r (Native.domain h)
    initial (SourceTensor.source_read decodeWord t r m schedule scratch)
    (SourceTensor.destination_read decodeWord t r m schedule scratch)
  let post := fun version => SourceBacking.memory r l (states version) wordByte m
  refine ⟨states,post,ran,frame,fun _ => rfl,?_,?_,?_⟩
  · intro version j hj address ha b hb
    change Tensor.byte (Tensor.output t address.role)
      (SourceBacking.memory r l (states version) wordByte m)
      (address.row*r.width address.role+address.column) b = _
    rw [SourceBacking.tensor_readback t r h l ins binding _ wordByte m j hj address ha b hb,
      values version j hj address ha]
    simp only [RawNative.kernelNativeTarget,nativeTarget,initial,WholeBacking.input_row]
  · intro version p outside
    exact SourceBacking.frame r l (states version) wordByte m p outside
  · change SourceBacking.memory r l (states Bool.false) wordByte m =
      SourceBacking.memory r l (states Bool.true) wordByte m
    unfold SourceBacking.memory
    rw [stores]

#axiomsClean whole_shared_native
#specNonCircular nativeTarget avoiding [FP.Structural.exec,Shared.original,Shared.candidate,
  SourceBacking.memory,WholeBacking.memory,Shared.kernel,SharedExecution.executed]
end
end VeriTile.Bench.Repairs.KvCacheFilling.TensorBridge.Native
