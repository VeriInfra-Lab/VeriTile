import bench.tritonbench_g.sgmv_expand_slice.Sources.Kernels
import bench.tritonbench_g.sgmv_expand_slice.Proofs.HostMetadata
import bench.tritonbench_g.bgmv_expand_slice.Sources.Contract

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Contract
open VeriTile.Torch
noncomputable section
open Classical
abbrev TensorInfo := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.TensorInfo
abbrev rows := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows
abbrev columns := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.columns
abbrev rank := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank
abbrev width := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width
abbrev Dense := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Dense
abbrev Singleton := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Singleton

structure Inputs where
  x : TensorInfo
  w : TensorInfo
  y : TensorInfo
  starts : TensorInfo
  lengths : TensorInfo
  indices : TensorInfo
structure Scalars where
  batches : Int
  maxSequence : Int
  tokens : Int
  offset : Int
  size : Int
  integerTypes : Bool
  addBool : Bool
  add : Bool
def tensors (t : Inputs) := [t.x,t.w,t.y,t.starts,t.lengths,t.indices]
def metadata (t : Inputs) := [t.starts,t.lengths,t.indices]
def reads (t : Inputs) := [t.x,t.w,t.starts,t.lengths,t.indices]
def shape (t : Inputs) (s : Scalars) : Shape :=
  ⟨s.tokens.toNat,rows t.y,columns t.y,rows t.w,width t.w,rank t.w,s.offset.toNat,s.batches.toNat⟩
def indexBytes (t : TensorInfo) : Nat := if t.dtype=.other "int64" then 8 else 4
def indexWidth (t : TensorInfo) : MetadataWidth := if t.dtype=.other "int64" then .w64 else .w32
def abi (t : Inputs) (s : Scalars) : SourceABI :=
  ⟨.single,.half,.single,indexWidth t.starts,indexWidth t.lengths,indexWidth t.indices,s.add⟩

/-- Ordinary signed tolist observations of original contiguous metadata.
Lengths are shape facts supplied by this local input API, not kernel results. -/
structure Lists (t : Inputs) where
  starts : List Int
  lengths : List Int
  indices : List Int
  startCount : starts.length=t.starts.numel
  lengthCount : lengths.length=t.lengths.numel
  indexCount : indices.length=t.indices.numel
def raw (t : Inputs) (s : Scalars) (lists : Lists t) :=
  HostMetadata.rawSegments s.batches.toNat lists.starts lists.lengths lists.indices
def segments (t : Inputs) (s : Scalars) (lists : Lists t) := (raw t s lists).map RawSegment.decode

def Layouts (t : Inputs) : Prop := ∀ tensor∈tensors t,tensor.layout=.strided
def DTypes (t : Inputs) : Prop :=
  t.x.dtype=.floating .fp32 ∧ t.w.dtype=.other "float16" ∧ t.y.dtype=.floating .fp32 ∧
  ∀ tensor∈metadata t,tensor.dtype=.other "int32" ∨ tensor.dtype=.other "int64"
def Ranks (t : Inputs) : Prop :=
  t.x.shape.length=2 ∧ t.y.shape.length=2 ∧ (t.w.shape.length=3 ∨ t.w.shape.length=4) ∧
  ∀ tensor∈metadata t,tensor.shape.length=1
def Devices (t : Inputs) : Prop :=
  ∀ tensor∈tensors t,tensor.device.kind="cuda" ∧ tensor.device=t.x.device
def ScalarTypes (s : Scalars) : Prop :=
  s.integerTypes=Bool.true ∧ s.addBool=Bool.true ∧ 0≤s.batches ∧ 0≤s.maxSequence ∧
  0≤s.tokens ∧ 0≤s.offset ∧ 0≤s.size
def Shapes (t : Inputs) (s : Scalars) : Prop :=
  t.x.shape=[s.tokens.toNat,rank t.w] ∧ s.size=(width t.w:Int) ∧
  s.offset+(width t.w:Int)≤columns t.y ∧
  t.starts.numel=s.batches.toNat ∧ t.indices.numel=s.batches.toNat ∧ s.batches.toNat≤t.lengths.numel
def Dimensions (t : Inputs) (s : Scalars) : Prop :=
  s.batches.toNat<2^31 ∧ s.tokens.toNat<2^31 ∧ rows t.y<2^31 ∧ rows t.w<2^31 ∧
  width t.w<2^31 ∧ rank t.w<2^31 ∧ columns t.y<2^31 ∧ width t.w*rank t.w<2^31 ∧ s.offset.toNat<2^31
def Extents (t : Inputs) : Prop :=
  4*t.x.numel<2^63 ∧ 2*t.w.numel<2^63 ∧ 4*t.y.numel<2^63 ∧
  ∀ tensor∈metadata t,tensor.numel*indexBytes tensor<2^63
/-- Observations of complete untyped_storage(), not merely the Tensor view.
No identity-disjointness rule is assumed. Zero-byte ranges are empty. -/
structure StorageRange where
  base : Nat
  bytes : Nat
  deriving DecidableEq, Repr
structure Backings where
  x : StorageRange
  w : StorageRange
  y : StorageRange
  starts : StorageRange
  lengths : StorageRange
  indices : StorageRange
  deriving DecidableEq, Repr

def StorageRange.covers (a : StorageRange) (address : Nat) :=
  a.base≤address ∧ address<a.base+a.bytes

def StorageRange.disjoint (a b : StorageRange) :=
  a.bytes=0 ∨ b.bytes=0 ∨ a.base+a.bytes≤b.base ∨ b.base+b.bytes≤a.base

def Separate (a : Backings) : Prop :=
  a.y.disjoint a.x ∧ a.y.disjoint a.w ∧ a.y.disjoint a.starts ∧ a.y.disjoint a.lengths ∧ a.y.disjoint a.indices

theorem disjoint_byte (a b : StorageRange) (h : a.disjoint b)
    (address : Nat) (ha : a.covers address) (hb : b.covers address) : False := by
  rcases h with h|h|h|h <;> simp only [StorageRange.covers] at ha hb <;> omega

theorem disjoint_iff_no_common_byte (a b : StorageRange) :
    a.disjoint b ↔ ∀ address,¬(a.covers address ∧ b.covers address) := by
  constructor
  · intro h address both; exact disjoint_byte a b h address both.1 both.2
  · intro h
    by_contra hn
    simp only [StorageRange.disjoint,not_or] at hn
    have both : a.covers (max a.base b.base) ∧ b.covers (max a.base b.base) := by
      simp only [StorageRange.covers]
      omega
    exact h _ both


def Aligned (t : Inputs) (s : Scalars) : Prop :=
  s.batches=0 ∨ width t.w=0 ∨ ∀ tensor∈[t.x,t.w,t.y],tensor.numel=0 ∨ tensor.dataPtr.toNat%16=0
def Content (t : Inputs) (s : Scalars) (lists : Lists t) : Prop :=
  ∀ segment∈raw t s lists,segment.accepted (shape t s) ∧
    (0<segment.length ∧ segment.adapter≠-1 → tiles segment.length.toNat*tiles (width t.w)≤2^31-1)

structure InputContract (t : Inputs) (s : Scalars) (spans : Backings) (lists : Lists t) : Prop where
  layouts : Layouts t
  dtypes : DTypes t
  ranks : Ranks t
  singleton : Singleton t.w
  devices : Devices t
  dense : ∀ tensor∈tensors t,Dense tensor
  scalarTypes : ScalarTypes s
  shapes : Shapes t s
  dimensions : Dimensions t s
  extents : Extents t
  separate : Separate spans
  aligned : Aligned t s
  content : Content t s lists

def pythonChecks (t : Inputs) (s : Scalars) (spans : Backings) (lists : Lists t) : List Bool :=
  [decide (Layouts t),decide (DTypes t),decide (Ranks t),decide (Singleton t.w),decide (Devices t),
   decide (∀ tensor∈tensors t,Dense tensor),decide (ScalarTypes s),decide (Shapes t s),decide (Dimensions t s),
   decide (Extents t),decide (Separate spans),decide (Aligned t s),decide (Content t s lists)]
def Admitted (t : Inputs) (s : Scalars) (spans : Backings) (lists : Lists t) := (pythonChecks t s spans lists).all id=Bool.true
theorem python_checks_iff (t : Inputs) (s : Scalars) (spans : Backings) (lists : Lists t) :
    Admitted t s spans lists ↔ InputContract t s spans lists := by
  simp only [Admitted,pythonChecks,List.all_cons,List.all_nil,Bool.and_eq_true,decide_eq_true_eq,id_eq,and_true]
  constructor
  · rintro ⟨a,b,c,d,e,f,g,h,j,k,l,m,n⟩;exact ⟨a,b,c,d,e,f,g,h,j,k,l,m,n⟩
  · intro h;exact ⟨h.layouts,h.dtypes,h.ranks,h.singleton,h.devices,h.dense,h.scalarTypes,h.shapes,h.dimensions,h.extents,h.separate,h.aligned,h.content⟩

theorem validated (ok : Admitted t s spans lists) :
    validateSegments (shape t s) (raw t s lists)=some (segments t s lists) := by
  apply (validateSegments_spec _ _ _).mpr
  exact ⟨fun segment member => ((python_checks_iff t s spans lists).mp ok).content segment member |>.1,rfl⟩

theorem fitted (ok : Admitted t s spans lists) : ∀ segment∈segments t s lists,SegmentFits (shape t s) segment :=
  (validated_segments_fit _ _ _ (validated ok)).2
theorem capacity (ok : Admitted t s spans lists) : (shape t s).slice+(shape t s).columns≤(shape t s).outputColumns := by
  have c:=(python_checks_iff t s spans lists).mp ok
  have hn:=c.scalarTypes.2.2.2.2.2.1
  have bound:=c.shapes.2.2.1
  change s.offset.toNat+width t.w≤columns t.y
  omega

#axiomsClean disjoint_byte
#axiomsClean disjoint_iff_no_common_byte
#axiomsClean python_checks_iff
#axiomsClean validated
#axiomsClean fitted
#axiomsClean capacity
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.Contract
