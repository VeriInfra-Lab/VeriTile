import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.Targets
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
open VeriTile.Triton FP.Structural TokenAttnMistral
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Shared caller Tensor objects retain allocation identity, storage offset,
physicalIndex and observed metadata. The container has no imposed dtype. -/
structure TensorCall where
  tensors : Role → VeriTile.Torch.FP32.Tensor
  slidingWindow : Int
  windowIsInt : Bool
instance : CoeFun TensorCall (fun _ => Role → VeriTile.Torch.FP32.Tensor) := ⟨TensorCall.tensors⟩

def TensorCall.metadata (c : TensorCall) : Inputs :=
  ⟨fun r => (c r).metadata,c.slidingWindow,c.windowIsInt⟩

/-- Ordinary four-byte Tensor API laws, for both FP32 and int32. Dtypes are
checked by Header. These laws concern storage only; they assume no arithmetic,
kernel execution, launch success or result, and impose no allocation separation. -/
structure TensorLaws (c : TensorCall) : Prop where
  inBounds : ∀ r i,i < (c r).metadata.numel → (c r).physicalIndex i < (c r).storage.elements
  pointer : ∀ r,0 < (c r).metadata.numel →
    (c r).metadata.dataPtr=BitVec.ofNat 64 ((c r).storage.address (c r).storageOffset)
  contiguous : ∀ r,(c r).metadata.isContiguous=Bool.true →
    ∀ i,i < (c r).metadata.numel → (c r).physicalIndex i=(c r).storageOffset+i

def logicalAddress (c : TensorCall) (r : Role) (i : Nat) :=
  (c r).storage.address ((c r).physicalIndex i)
def logical {α : Type} (read : Nat → α) (c : TensorCall) (r : Role) (i : Nat) :=
  read (logicalAddress c r i)

theorem tensor_address (c : TensorCall) (laws : TensorLaws c) (header : Header c.metadata)
    (r : Role) (i : Nat) (hi : i < (c r).metadata.numel) :
    logicalAddress c r i=base c.metadata r+4*i := by
  have contig := (header.ordinary r).2
  have zero := laws.inBounds r 0 (by omega)
  rw [laws.contiguous r contig 0 (by omega)] at zero
  have span := (c r).storage.spanFits
  have bound : (c r).storage.address (c r).storageOffset < 2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address]
    omega
  unfold logicalAddress base TensorCall.metadata
  rw [laws.contiguous r contig i hi,laws.pointer r (by omega)]
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound,VeriTile.Torch.FP32.Storage.address]
  omega

theorem logical_read {α : Type} (read : Nat → α) (c : TensorCall) (laws : TensorLaws c)
    (header : Header c.metadata) (r : Role) (i : Nat) (hi : i < (c r).metadata.numel) :
    logical read c r i=read (base c.metadata r+4*i) := by
  exact congrArg read (tensor_address c laws header r i hi)

theorem shape_two (m : VeriTile.Torch.TensorMetadata) (rank : m.shape.length=2) :
    m.shape=[m.shape.headD 0,m.shape[1]?.getD 0] := by
  cases hs : m.shape with
  | nil => simp [hs] at rank
  | cons a rest => cases rest with
    | nil => simp [hs] at rank
    | cons b rest => cases rest with
      | nil => simp [hs]
      | cons c rest => simp [hs] at rank

theorem shape_three (m : VeriTile.Torch.TensorMetadata) (rank : m.shape.length=3) :
    m.shape=[m.shape.headD 0,m.shape[1]?.getD 0,m.shape[2]?.getD 0] := by
  cases hs : m.shape with
  | nil => simp [hs] at rank
  | cons a rest => cases rest with
    | nil => simp [hs] at rank
    | cons b rest => cases rest with
      | nil => simp [hs] at rank
      | cons c rest => cases rest with
        | nil => simp [hs]
        | cons d rest => simp [hs] at rank

theorem extents (m : Inputs) (header : Header m) :
    (m .probability).numel=heads m*tokens m ∧
    (m .value).numel=cache m*kvheads m*width m ∧
    (m .pages).numel=requests m*columns m := by
  have p := shape_two _ header.ranks.1
  have v := shape_three _ header.ranks.2.1
  have t := shape_two _ header.ranks.2.2.2.1
  constructor
  · unfold VeriTile.Torch.TensorMetadata.numel
    rw [p]
    simp [heads,tokens]
  constructor
  · unfold VeriTile.Torch.TensorMetadata.numel
    rw [v]
    simp [cache,kvheads,width,Nat.mul_assoc]
  · unfold VeriTile.Torch.TensorMetadata.numel
    rw [t]
    simp [requests,columns]

theorem data_bounds (m : Inputs) (read : Reader) (contract : InputContract m read)
    (b : Nat) (hb : b < batches m) :
    index m read .request b < requests m ∧ index m read .sequence b ≤ columns m ∧
    index m read .start b+index m read .length b ≤ tokens m ∧
    ∀ n,first m read b+n < index m read .sequence b → page m read b n < cache m := by
  obtain ⟨hr,hl,hs,hp⟩ := contract.contents b hb
  simp only [index,page]
  constructor
  · omega
  constructor
  · omega
  constructor
  · omega
  · intro n hn
    have h := hp n hn
    simp only [index] at h
    omega

theorem group_shape (m : Inputs) (header : Header m) :
    0 < group m ∧ kvheads m*group m=heads m := by
  have mul : group m*kvheads m=heads m :=
    Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero header.headShape.2.2)
  have pos := header.headShape.1
  constructor
  · nlinarith
  · simpa only [Nat.mul_comm] using mul

theorem group_bound (m : Inputs) (header : Header m) (h : Nat) (hh : h < heads m) :
    h/group m < kvheads m := by
  rw [Nat.div_lt_iff_lt_mul (group_shape m header).1]
  simpa [(group_shape m header).2] using hh

/-- The probability load follows the attention-length mask alone. -/
theorem probability_bound (m : Inputs) (read : Reader) (contract : InputContract m read)
    (b h n : Nat) (hb : b < batches m) (hh : h < heads m)
    (hn : n < index m read .length b) :
    probabilityIndex m read b h n < (m .probability).numel := by
  obtain ⟨_,_,hs,_⟩ := data_bounds m read contract b hb
  rw [(extents m contract.header).1]
  exact linear_bound _ _ _ _ hh (by omega)

/-- Table and V use the independent sequence-window mask, even beyond the
probability tail. The checked whole suffix supplies these bounds. -/
theorem value_bounds (m : Inputs) (read : Reader) (contract : InputContract m read)
    (b h n d : Nat) (hb : b < batches m) (hh : h < heads m)
    (hn : first m read b+n < index m read .sequence b) (hd : d < width m) :
    index m read .request b*columns m+(first m read b+n) < (m .pages).numel ∧
    valueIndex m read b h n d < (m .value).numel := by
  obtain ⟨hr,hl,_,hp⟩ := data_bounds m read contract b hb
  obtain ⟨_,ve,te⟩ := extents m contract.header
  constructor
  · rw [te]
    exact linear_bound _ _ _ _ hr (by omega)
  · rw [ve]
    have bound := linear_bound _ _ _ _
      (linear_bound _ _ _ _ (hp n hn) (group_bound m contract.header h hh)) hd
    simpa [valueIndex,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using bound

theorem probability_logical {α : Type} (c : TensorCall) (laws : TensorLaws c)
    (metadata : Reader) (read : Nat → α) (contract : InputContract c.metadata metadata)
    (b h n : Nat) (hb : b < batches c.metadata) (hh : h < heads c.metadata)
    (hn : n < index c.metadata metadata .length b) :
    read (base c.metadata .probability+4*probabilityIndex c.metadata metadata b h n)=
      logical read c .probability (probabilityIndex c.metadata metadata b h n) :=
  (logical_read read c laws contract.header _ _
    (probability_bound c.metadata metadata contract b h n hb hh hn)).symm

theorem value_logical {α : Type} (c : TensorCall) (laws : TensorLaws c)
    (metadata : Reader) (read : Nat → α) (contract : InputContract c.metadata metadata)
    (b h n d : Nat) (hb : b < batches c.metadata) (hh : h < heads c.metadata)
    (hn : first c.metadata metadata b+n < index c.metadata metadata .sequence b)
    (hd : d < width c.metadata) :
    read (base c.metadata .value+4*valueIndex c.metadata metadata b h n d)=
      logical read c .value (valueIndex c.metadata metadata b h n d) :=
  (logical_read read c laws contract.header _ _
    (value_bounds c.metadata metadata contract b h n d hb hh hn hd).2).symm

theorem gather_logical (c : TensorCall) (laws : TensorLaws c) (metadata : Reader)
    (contract : InputContract c.metadata metadata) (b h n d : Nat)
    (hb : b < batches c.metadata) (hh : h < heads c.metadata)
    (hn : first c.metadata metadata b+n < index c.metadata metadata .sequence b)
    (hd : d < width c.metadata) :
    page c.metadata metadata b n=
      (logical metadata c .pages (index c.metadata metadata .request b*columns c.metadata+
        (first c.metadata metadata b+n))).toInt.toNat := by
  rw [logical_read metadata c laws contract.header _ _
    (value_bounds c.metadata metadata contract b h n d hb hh hn hd).1]
  rfl

def logicalRealTarget (c : TensorCall) (metadata : Reader) (read : Nat → ℝ) (b h d : Nat) : ℝ :=
  ∑ n ∈ Finset.range (index c.metadata metadata .length b),
    if first c.metadata metadata b+n < index c.metadata metadata .sequence b then
    logical read c .probability (probabilityIndex c.metadata metadata b h n)*
    logical read c .value (valueIndex c.metadata metadata b h n d)
    else 0

theorem real_logical (c : TensorCall) (laws : TensorLaws c) (metadata : Reader)
    (read : Nat → ℝ) (contract : InputContract c.metadata metadata)
    (b h d : Nat) (hb : b < batches c.metadata) (hh : h < heads c.metadata)
    (hd : d < width c.metadata) :
    realTarget c.metadata metadata read b h d=logicalRealTarget c metadata read b h d := by
  unfold realTarget logicalRealTarget
  apply Finset.sum_congr rfl
  intro n hn
  split_ifs with hv
  · rw [probability_logical c laws metadata read contract b h n hb hh (Finset.mem_range.mp hn),
      value_logical c laws metadata read contract b h n d hb hh hv hd]
  · rfl

/-- Physical separation is checked for used views, not whole allocations. -/
theorem read_exterior (m : Inputs) (metadata : Reader) (contract : InputContract m metadata)
    (live : 0 < batches m) (r : Role) (hr : r∈reads) (i : Nat) (hi : i < (m r).numel) :
    Exterior m (base m r+4*i) := by
  have apart := (contract.physical live).output r hr
  have outPositive : 0 < (m .output).numel := by
    simp only [VeriTile.Torch.TensorMetadata.numel,contract.header.outputShape,
      List.prod_cons,List.prod_nil,Nat.mul_one]
    have dim := contract.header.dimension
    have heads := contract.header.headShape.1
    have positive : 0 < width m := by rcases dim with hd|hd|hd|hd <;> omega
    exact Nat.mul_pos live (Nat.mul_pos heads positive)
  unfold Apart at apart
  unfold Exterior
  omega

#axiomsClean tensor_address
#axiomsClean probability_bound
#axiomsClean value_bounds
#axiomsClean probability_logical
#axiomsClean value_logical
#axiomsClean gather_logical
#axiomsClean real_logical
#axiomsClean read_exterior
#specNonCircular logicalRealTarget avoiding [kernel,FP.Structural.exec,Grid.run,outputValues]
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
