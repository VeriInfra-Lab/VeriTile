import bench.repairs.tritonbench_g.token_attn_mistral.Integers
import bench.repairs.tritonbench_g.token_attn_mistral.HostSchedule
namespace VeriTile.Bench.Repairs.TokenAttnMistral.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 4000000

/-- Actual signed int32 payloads before the Nat index adapter. The binding
below is the ordinary Tensor/read interpretation, never a result premise. -/
structure SignedMetadata where
  (request start length sequence : Nat → Int)
  page : Nat → Nat → Int

structure SignedBinding {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata)
    (m : Memory α) : Prop where
  request : ∀ b<batches c, requestAt c m b=(raw.request b).toNat
  start : ∀ b<batches c, startAt c m b=(raw.start b).toNat
  length : ∀ b<batches c, seq c m b=(raw.length b).toNat
  sequence : ∀ b<batches c, fullLength c m b=(raw.sequence b).toNat
  page : ∀ b<batches c, ∀ n, n+windowAt c m b<fullLength c m b →
    pageAt c m b n=(raw.page (requestAt c m b) (windowAt c m b+n)).toNat

/-- Header/runtime observations; all tensor spans are rebased to the current
contiguous view (offset zero, capacity=numel). This does not copy payloads. -/
structure Flags where
  cuda : Bool
  (negative contiguous : Tensor → Bool)

/-- Explicit checked admission. Data contains integer bounds and storage-tag
facts only. FP32 payloads have no finiteness, normalization or value test.
Layout/Storage are the descriptor and view-span checks; their consequences
include every active access and all padded source index expressions. -/
structure Admission {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata)
    (m : Memory α) (flags : Flags) : Prop where
  cuda : flags.cuda=Bool.true
  ordinary : ∀ t∈[c.probability.toTensor,c.value.toTensor,c.output.toTensor,c.pages.toTensor,c.request.toTensor,c.start.toTensor,c.length.toTensor,c.sequence.toTensor,c.unusedStart.toTensor], flags.negative t=Bool.false ∧ flags.contiguous t=Bool.true
  layout : Layout c
  data : Data c m
  storage : Storage c
  integers : IntegerDomain c
  dimension : c.value.width=16 ∨ c.value.width=32 ∨ c.value.width=64 ∨ c.value.width=128
  probabilityPad : heads c*c.probability.cols+128<2^31
  probabilityDense : c.probability.rowStride=c.probability.cols ∧ c.probability.colStride=1
  vectors : ∀ t∈[c.request,c.start,c.length,c.sequence,c.unusedStart], t.size=batches c ∧ t.stride=1
  signed : ∀ b<batches c, 0≤raw.request b ∧ 0≤raw.start b ∧ 0≤raw.length b ∧ 0≤raw.sequence b
  signedPages : ∀ b<batches c, ∀ n, n+windowAt c m b<fullLength c m b →
    0≤raw.page (requestAt c m b) (windowAt c m b+n)
  mixedApart : ∀ f, f∈[c.probability.toTensor,c.value.toTensor] → ∀ i, i∈[c.pages.toTensor,c.request.toTensor,c.start.toTensor,c.length.toTensor,c.sequence.toTensor,c.unusedStart.toTensor] →
    f.capacity=0 ∨ i.capacity=0 ∨ f.allocationBase+4*f.capacity ≤ i.allocationBase ∨ i.allocationBase+4*i.capacity ≤ f.allocationBase
  alignment : ∀ t∈[c.probability.toTensor,c.value.toTensor,c.output.toTensor,c.pages.toTensor,c.request.toTensor,c.start.toTensor,c.length.toTensor,c.sequence.toTensor,c.unusedStart.toTensor], 0<t.capacity → byteAddress t 0%4=0
  backing : ∀ t∈[c.probability.toTensor,c.value.toTensor,c.output.toTensor,c.pages.toTensor,c.request.toTensor,c.start.toTensor,c.length.toTensor,c.sequence.toTensor,c.unusedStart.toTensor], t.allocationBase+4*t.capacity≤2^64

/-- Declarative finite mirror of the Python guard, under ordinary descriptor
and current-storage binding. No execution or output appears in the decision. -/
def check {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata) (m : Memory α) (flags : Flags) : Bool :=
  @decide (Admission c raw m flags) (Classical.propDecidable _)

theorem check_iff {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata) (m : Memory α) (flags : Flags) :
    check c raw m flags=Bool.true ↔ Admission c raw m flags := by simp [check]

/-- Checked nonnegativity makes signed metadata recovery exact. -/
theorem signed_metadata {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata)
    (m : Memory α) (flags : Flags) (a : Admission c raw m flags)
    (binding : SignedBinding c raw m) (b : Nat) (hb : b<batches c) :
    (requestAt c m b : Int)=raw.request b ∧ (startAt c m b : Int)=raw.start b ∧
      (seq c m b : Int)=raw.length b := by
  rw [binding.request b hb,binding.start b hb,binding.length b hb]
  exact ⟨Int.toNat_of_nonneg (a.signed b hb).1,
    Int.toNat_of_nonneg (a.signed b hb).2.1,
    Int.toNat_of_nonneg (a.signed b hb).2.2.1⟩

/-- P/page/V addresses are formed before the first loop test, including
zero-length rows. Their int32 range follows from the same checked extents. -/
theorem formed_prelude {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata)
    (m : Memory α) (flags : Flags) (a : Admission c raw m flags)
    (b h lane d : Nat) (hb : b<batches c) (hh : h<heads c)
    (hl : lane<128) (hd : d<c.value.width) :
    h*c.probability.cols+(startAt c m b+lane)<2^31 ∧
    requestAt c m b*c.pages.cols+(windowAt c m b+lane)<2^31 ∧
    (h/group c)*c.value.width+d<2^31 ∧
    (b*heads c+h)*c.value.width+d<2^31 := by
  have start := a.data.probabilityWindow b hb
  have hp := Nat.mul_le_mul_right c.probability.cols (show h+1≤heads c by omega)
  have prob := a.probabilityPad
  have window : windowAt c m b≤c.pages.cols :=
    (Nat.sub_le _ _).trans (a.data.lengths b hb)
  have req := Nat.mul_le_mul_right c.pages.cols (show requestAt c m b+1≤c.pages.rows from a.data.requests b hb)
  have table := a.integers.table
  have outer := linear_bound b (batches c) h (heads c) hb hh
  have output := linear_bound _ (batches c*heads c) d c.value.width outer hd
  have value := linear_bound (h/group c) (heads c) d c.value.width
    (lt_of_le_of_lt (Nat.div_le_self _ _) hh) hd
  have nonempty := Nat.mul_le_mul_right (heads c*c.value.width) (show 1≤batches c by omega)
  have cap := a.integers.output
  simp only [Nat.add_mul,Nat.one_mul] at hp req
  simp only [Nat.one_mul] at nonempty
  rw [← Nat.mul_assoc] at nonempty
  exact ⟨by omega,by omega,by omega,by omega⟩

/-- Every byte of an active word is in the physical backing and does not wrap. -/
theorem complete_word (t : Tensor) (i : Nat) (live : InStorage t i)
    (backing : t.allocationBase+4*t.capacity≤2^64) (j : Nat) (hj : j<4) :
    byteAddress t i+j<t.allocationBase+4*t.capacity ∧
      (BitVec.ofNat 64 (byteAddress t i+j)).toNat=byteAddress t i+j := by
  have h:=live.2.2
  have bound : byteAddress t i+j<2^64 := by omega
  exact ⟨by omega,by change _ % 2^64 = _; exact Nat.mod_eq_of_lt bound⟩

/-- Observe the result in the caller's existing physical byte heap. This is
semantic readback of direct stores, not a Python allocation or copyback. -/
def physical {α : Type} (encode : Cell α → Fin 4 → BitVec 8) (c : Call)
    (before : Nat → BitVec 8) (after : Memory α) (address : Nat) : BitVec 8 :=
  let base := byteAddress c.output.toTensor 0
  if base≤address ∧ address<base+4*(batches c*heads c*c.value.width) then
    encode (after c.output.region ((address-base)/4)) ⟨(address-base)%4,Nat.mod_lt _ (by decide)⟩
  else before address

theorem physical_frame {α : Type} (encode : Cell α → Fin 4 → BitVec 8) (c : Call)
    (before : Nat → BitVec 8) (after : Memory α) (address : Nat)
    (outside : address<byteAddress c.output.toTensor 0 ∨
      byteAddress c.output.toTensor 0+4*(batches c*heads c*c.value.width)≤address) :
    physical encode c before after address=before address := by
  simp only [physical]
  rw [if_neg (by omega)]

/-- No native pointer, alignment or alias condition is imposed when B=0. -/
structure EmptyAdmission {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata)
    (m : Memory α) (flags : Flags) : Prop where
  empty : batches c=0
  cuda : flags.cuda=Bool.true
  ordinary : ∀ t∈[c.probability.toTensor,c.value.toTensor,c.output.toTensor,c.pages.toTensor,c.request.toTensor,c.start.toTensor,c.length.toTensor,c.sequence.toTensor,c.unusedStart.toTensor], flags.negative t=Bool.false ∧ flags.contiguous t=Bool.true
  layout : Layout c
  data : Data c m
  integers : IntegerDomain c
  dimension : c.value.width=16 ∨ c.value.width=32 ∨ c.value.width=64 ∨ c.value.width=128
  probabilityPad : heads c*c.probability.cols+128<2^31
  probabilityDense : c.probability.rowStride=c.probability.cols ∧ c.probability.colStride=1
  vectors : ∀ t∈[c.request,c.start,c.length,c.sequence,c.unusedStart], t.size=batches c ∧ t.stride=1
  signed : ∀ b<batches c, 0≤raw.request b ∧ 0≤raw.start b ∧ 0≤raw.length b ∧ 0≤raw.sequence b
  signedPages : ∀ b<batches c, ∀ n, n+windowAt c m b<fullLength c m b →
    0≤raw.page (requestAt c m b) (windowAt c m b+n)

def validate {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata) (m : Memory α) (flags : Flags) : Except Unit Bool := by
  classical
  exact if batches c=0 then
    if EmptyAdmission c raw m flags then .ok Bool.false else .error ()
  else if Admission c raw m flags then .ok Bool.true else .error ()

theorem validate_launch {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata) (m : Memory α) (flags : Flags) :
    validate c raw m flags=.ok Bool.true ↔ 0<batches c ∧ Admission c raw m flags := by
  classical
  unfold validate
  split_ifs <;> simp_all <;> omega

theorem validate_empty {α : Type} [Inhabited α] (c : Call) (raw : SignedMetadata) (m : Memory α) (flags : Flags) :
    validate c raw m flags=.ok Bool.false ↔ EmptyAdmission c raw m flags := by
  classical
  unfold validate
  split_ifs <;> simp_all
  all_goals
    intro h
    have e:=h.empty
    omega

/-- Checked wrapper control flow, including the exact successful empty return. -/
def dispatch {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (c : Call) (raw : SignedMetadata) (guardMemory : Memory β) [Inhabited β] (flags : Flags)
    (work : List PID) (m : Memory α) : Except Unit (Option (Result α)) := by
  classical
  exact if validate c raw guardMemory flags=.ok Bool.false then .ok (some ⟨none,c.output,m⟩)
    else if validate c raw guardMemory flags=.ok Bool.true then .ok (invokeActual M R c work m)
    else .error ()

theorem dispatch_empty {α β : Type} [Inhabited α] [Inhabited β]
    (M : Algebra α) (R : SumPrimitive α) (c : Call) (raw : SignedMetadata)
    (guardMemory : Memory β) (flags : Flags) (work : List PID) (m : Memory α)
    (h : validate c raw guardMemory flags=.ok Bool.false) :
    dispatch M R c raw guardMemory flags work m=.ok (some ⟨none,c.output,m⟩) := by simp [dispatch,h]

theorem dispatch_launch {α β : Type} [Inhabited α] [Inhabited β]
    (M : Algebra α) (R : SumPrimitive α) (c : Call) (raw : SignedMetadata)
    (guardMemory : Memory β) (flags : Flags) (work : List PID) (m : Memory α)
    (h : validate c raw guardMemory flags=.ok Bool.true) :
    dispatch M R c raw guardMemory flags work m=.ok (invokeActual M R c work m) := by simp [dispatch,h]

/-- Successful empty wrapper branch, preserving noncanonical empty layouts. -/
def emptyCall {α : Type} (c : Call) (memory : α) := ((none : Option Nat),c.output,memory)
theorem empty_call {α : Type} (c : Call) (memory : α) :
    (emptyCall c memory).1=none ∧ (emptyCall c memory).2.1=c.output ∧
      (emptyCall c memory).2.2=memory := ⟨rfl,rfl,rfl⟩

#axiomsClean formed_prelude
#axiomsClean signed_metadata
#axiomsClean complete_word
#axiomsClean physical_frame
#specNonCircular Admission avoiding [FP.Structural.exec,Grid.run,invoke,invokeActual]
end
end VeriTile.Bench.Repairs.TokenAttnMistral.Host
