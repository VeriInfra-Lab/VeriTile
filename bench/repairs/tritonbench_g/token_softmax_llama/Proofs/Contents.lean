import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.TensorHost

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Triton FP.Structural Strided.Grid
noncomputable section

theorem check_input_iff (c : Call) (s n : Nat → Int) :
    checkInput c s n = .ok () ↔ InputContract c s n := by
  have shape : InputContract c s n ↔ MetadataContract c ∧ (contentActive c → ContentChecks c s n) :=
    ⟨fun h => ⟨h.metadata,h.contents⟩,fun h => ⟨h.1,h.2⟩⟩
  rw [shape,←check_iff]
  unfold checkInput checkContents
  cases result : check c <;>
    by_cases a : contentActive c <;> by_cases b : ContentChecks c s n <;> simp_all

theorem check_call_arguments_iff (x s n o : Argument) (k : Maximum) (starts lengths : Nat → Int) :
    checkCallArguments x s n o k starts lengths = .ok () ↔
      AdmittedCallArguments x s n o k starts lengths := by
  cases x <;> cases s <;> cases n <;> cases o <;> cases k <;>
    simp [checkCallArguments,AdmittedCallArguments]
  split_ifs <;> simp_all [check_input_iff]

def observedStart (c : TensorCall) (raw : RegionName → Nat → BitVec 32) (b : Nat) : Int :=
  (raw (storageRegion c.start.storage) (c.start.storageOffset+b*c.start.stride)).toInt
def observedLength (c : TensorCall) (raw : RegionName → Nat → BitVec 32) (b : Nat) : Int :=
  (raw (storageRegion c.length.storage) (c.length.storageOffset+b*c.length.stride)).toInt

/-- Ordinary cell typing of the supplied FP32 storage, independent of values. -/
def PayloadCells {α} [Inhabited α] (c : TensorCall) (m : Memory α) : Prop :=
  ∀ head < rows c.input.observation, ∀ token < cols c.input.observation,
    ∃ x, (view c m) "input" (head*c.input.rowStride+token*c.input.colStride)=.mk .fp32 x

def invokeChecked {α} [Inhabited α] (algebra : Algebra α) (c : TensorCall)
    (work : List PID) (raw : RegionName → Nat → BitVec 32) (m : Memory α) :=
  if checkInput c.observation (observedStart c raw) (observedLength c raw) = .ok () then
    invoke algebra c work m else none

theorem invoke_checked {α} [Inhabited α] (algebra : Algebra α) (c : TensorCall)
    (work : List PID) (raw : RegionName → Nat → BitVec 32) (m : Memory α)
    (h : InputContract c.observation (observedStart c raw) (observedLength c raw)) :
    invokeChecked algebra c work raw m = invoke algebra c work m := by
  simp [invokeChecked,(check_input_iff _ _ _).mpr h]

/-- Every admitted no-launch path retains the exact supplied Tensor and memory. -/
theorem no_launch_identity {α} [Inhabited α] (algebra : Algebra α) (c : TensorCall)
    (work : List PID) (raw : RegionName → Nat → BitVec 32) (m : Memory α)
    (h : InputContract c.observation (observedStart c raw) (observedLength c raw))
    (inactive : ¬active c.observation) :
    invokeChecked algebra c work raw m = some ⟨none,c.output.tensor,m⟩ := by
  rw [invoke_checked _ _ _ _ _ h]
  simp [invoke,inactive]

theorem checked_data {α} [Inhabited α] (c : TensorCall)
    (raw : RegionName → Nat → BitVec 32) (m : Memory α)
    (h : InputContract c.observation (observedStart c raw) (observedLength c raw))
    (launch : contentActive c.observation) (pin : MetadataPin c raw m) (payload : PayloadCells c m) :
    Strided.Host.Data (host c.observation) (view c m) := by
  have hs := h.metadata.tensors.2.2.2.2.2.2.2.2.2.2.1
  have lengths : rows c.length.observation = rows c.start.observation := by
    change c.length.metadata.shape.headD 0 = c.start.metadata.shape.headD 0
    exact congrArg (fun shape : List Nat => shape.headD 0) hs
  have loaded (b : Nat) (hb : b < rows c.start.observation) :=
    metadata_loaded c raw m pin b hb (by rw [lengths];exact hb)
  have bounds (b : Nat) (hb : b < rows c.start.observation) :
      0 ≤ observedStart c raw b ∧ 0 ≤ observedLength c raw b ∧
      observedLength c raw b ≤ (c.maximum : Int) ∧
      observedStart c raw b + observedLength c raw b ≤ (cols c.input.observation : Int) :=
    (h.contents launch).1 b (List.mem_range.mpr hb)
  constructor
  · intro b hb
    rw [(loaded b hb).2]
    have := bounds b hb
    change (observedLength c raw b).toNat ≤ c.maximum
    omega
  · intro b hb
    rw [(loaded b hb).1,(loaded b hb).2]
    have := bounds b hb
    change (observedStart c raw b).toNat+(observedLength c raw b).toNat ≤ cols c.input.observation
    omega
  · intro b hb d hd ne bn dn
    rw [(loaded b hb).1,(loaded b hb).2,(loaded d hd).1,(loaded d hd).2]
    rw [(loaded b hb).2] at bn
    rw [(loaded d hd).2] at dn
    have bb := bounds b hb
    have dd := bounds d hd
    have apart := (h.contents launch).2 b (List.mem_range.mpr hb) d (List.mem_range.mpr hd) ne
    change 0 < (observedLength c raw b).toNat at bn
    change 0 < (observedLength c raw d).toNat at dn
    change (observedStart c raw b).toNat+(observedLength c raw b).toNat ≤
      (observedStart c raw d).toNat ∨
      (observedStart c raw d).toNat+(observedLength c raw d).toNat ≤
      (observedStart c raw b).toNat
    have := apart (by omega) (by omega)
    omega
  · intro b hb
    rw [(loaded b hb).1]
    have limit := BitVec.toInt_lt (x := raw (storageRegion c.start.storage)
      (c.start.storageOffset+b*c.start.stride))
    norm_num at limit ⊢
    omega
  · intro b hb
    rw [(loaded b hb).2]
    have limit := BitVec.toInt_lt (x := raw (storageRegion c.length.storage)
      (c.length.storageOffset+b*c.length.stride))
    norm_num at limit ⊢
    omega
  · intro b hb
    have start := pin.starts b hb
    have length := pin.lengths b (by rw [lengths];exact hb)
    rw [(loaded b hb).1,(loaded b hb).2]
    constructor
    · simpa [host,vector,descriptor,TensorCall.observation,MetadataTensor.observation,rs,view] using start
    · simpa [host,vector,descriptor,TensorCall.observation,MetadataTensor.observation,rs,view] using length
  · exact payload

#axiomsClean check_input_iff
#axiomsClean check_call_arguments_iff
#axiomsClean no_launch_identity
#axiomsClean checked_data
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
