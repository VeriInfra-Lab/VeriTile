import bench.repairs.tritonbench_g.token_attn_mistral.HostCore
namespace VeriTile.Bench.Repairs.TokenAttnMistral.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

/-- Capacity counts four-byte elements, relative to the allocation base. -/
def Fits (t : Tensor) (extent : Nat) := t.storageOffset+extent≤t.capacity
def byteAddress (t : Tensor) (index : Nat) := t.allocationBase+4*(t.storageOffset+index)

/-- Exact element span, including zero-size views and zero/nonunit strides. -/
def vectorExtent (size stride : Nat) := if size=0 then 0 else (size-1)*stride+1
def matrixExtent (rows cols rowStride colStride : Nat) :=
  if rows=0 ∨ cols=0 then 0 else (rows-1)*rowStride+(cols-1)*colStride+1

/-- Ordinary supplied tensor storage. The caller supplies the output; no
allocation or execution success is assumed. Physical output/input separation
is explicit in addition to the logical region separation in Layout. -/
structure Storage (c : Call) : Prop where
  probability : Fits c.probability.toTensor (matrixExtent (heads c) c.probability.cols c.probability.rowStride c.probability.colStride)
  value : Fits c.value.toTensor (c.value.slots*c.value.heads*c.value.width)
  output : Fits c.output.toTensor (batches c*heads c*c.value.width)
  pages : Fits c.pages.toTensor (c.pages.rows*c.pages.cols)
  request : Fits c.request.toTensor (vectorExtent c.request.size c.request.stride)
  start : Fits c.start.toTensor (vectorExtent c.start.size c.start.stride)
  length : Fits c.length.toTensor (vectorExtent c.length.size c.length.stride)
  sequence : Fits c.sequence.toTensor (vectorExtent c.sequence.size c.sequence.stride)
  unused : Fits c.unusedStart.toTensor (vectorExtent c.unusedStart.size c.unusedStart.stride)
  physicalSeparate : ∀ t∈[c.probability.toTensor,c.value.toTensor,c.pages.toTensor,
      c.request.toTensor,c.start.toTensor,c.length.toTensor,c.sequence.toTensor,c.unusedStart.toTensor],
    byteAddress c.output.toTensor (batches c*heads c*c.value.width)≤byteAddress t 0 ∨
    t.allocationBase+4*t.capacity≤byteAddress c.output.toTensor 0

def InStorage (t : Tensor) (i : Nat) :=
  t.storageOffset+i<t.capacity ∧
  t.allocationBase≤byteAddress t i ∧
  byteAddress t i+4≤t.allocationBase+4*t.capacity

theorem fits_address (t : Tensor) (n i : Nat) (fits : Fits t n) (hi : i<n) :
    InStorage t i := by
  simp only [Fits] at fits
  simp only [InStorage,byteAddress]
  omega

theorem linear_bound (a n b q : Nat) (ha : a<n) (hb : b<q) : a*q+b<n*q := by
  have step := Nat.mul_le_mul_right q (show a+1≤n by omega)
  nlinarith

theorem vector_bound (size stride i : Nat) (hi : i<size) :
    i*stride<vectorExtent size stride := by
  have hne : size≠0 := by omega
  have h := Nat.mul_le_mul_right stride (show i≤size-1 by omega)
  simp only [vectorExtent,if_neg hne]
  omega

theorem matrix_bound (rows cols rs cs i j : Nat) (hi : i<rows) (hj : j<cols) :
    i*rs+j*cs<matrixExtent rows cols rs cs := by
  have hne : ¬(rows=0 ∨ cols=0) := by omega
  have hr := Nat.mul_le_mul_right rs (show i≤rows-1 by omega)
  have hc := Nat.mul_le_mul_right cs (show j≤cols-1 by omega)
  simp only [matrixExtent,if_neg hne]
  omega

theorem group_bound (c : Call) (layout : Layout c) (h : Nat) (hh : h<heads c) :
    h/group c<c.value.heads := by
  apply (Nat.div_lt_iff_lt_mul layout.groupPositive).mpr
  simpa only [← layout.groupShape] using hh

/-- Exact active metadata, page, probability, value and output addresses.
Masked tail lanes have no loads and therefore do not require valid addresses. -/
structure Accesses {α : Type} [Inhabited α] (c : Call) (m : Memory α) : Prop where
  metadata : ∀ b<batches c,
    InStorage c.request.toTensor (b*c.request.stride) ∧
    InStorage c.start.toTensor (b*c.start.stride) ∧
    InStorage c.length.toTensor (b*c.length.stride) ∧
    InStorage c.sequence.toTensor (b*c.sequence.stride)
  probability : ∀ b<batches c, ∀ h<heads c, ∀ n<seq c m b,
    InStorage c.probability.toTensor (h*c.probability.rowStride+(startAt c m b+n)*c.probability.colStride)
  gathered : ∀ b<batches c, ∀ h<heads c, ∀ n, n+windowAt c m b<fullLength c m b → ∀ d<c.value.width,
    InStorage c.pages.toTensor (requestAt c m b*c.pages.rowStride+(windowAt c m b+n)*c.pages.colStride) ∧
    InStorage c.value.toTensor (pageAt c m b n*c.value.slotStride+(h/group c)*c.value.headStride+d*c.value.dimStride)
  output : ∀ b<batches c, ∀ h<heads c, ∀ d<c.value.width,
    InStorage c.output.toTensor (b*c.output.slotStride+h*c.output.headStride+d*c.output.dimStride)

theorem accesses {α : Type} [Inhabited α] (c : Call) (m : Memory α)
    (layout : Layout c) (data : Data c m) (storage : Storage c) : Accesses c m := by
  constructor
  · intro b hb
    exact ⟨fits_address _ _ _ storage.request (vector_bound _ _ _ hb),
      fits_address _ _ _ storage.start (vector_bound _ _ _ (lt_of_lt_of_le hb layout.metadataShape.1)),
      fits_address _ _ _ storage.length (vector_bound _ _ _ (lt_of_lt_of_le hb layout.metadataShape.2)),
      fits_address _ _ _ storage.sequence (vector_bound _ _ _ (lt_of_lt_of_le hb layout.sequenceShape))⟩
  · intro b hb h hh n hn
    exact fits_address _ _ _ storage.probability
      (matrix_bound _ _ _ _ _ _ hh (lt_of_lt_of_le (Nat.add_lt_add_left hn _) (data.probabilityWindow b hb)))
  · intro b hb h hh n hn d hd
    simp only [layout.pagesDense.1,layout.pagesDense.2,
      layout.valueDense.1,layout.valueDense.2.1,layout.valueDense.2.2,Nat.mul_one]
    constructor
    · apply fits_address _ _ _ storage.pages
      have fullbound := data.lengths b hb
      have tokenbound : windowAt c m b+n<c.pages.cols :=
        lt_of_lt_of_le (by simpa only [Nat.add_comm] using hn) fullbound
      exact linear_bound _ _ _ _ (data.requests b hb) tokenbound
    · apply fits_address _ _ _ storage.value
      have hi := linear_bound _ _ _ _ (data.pages b hb n hn) (group_bound c layout h hh)
      have all := linear_bound _ _ _ _ hi hd
      nlinarith
  · intro b hb h hh d hd
    simp only [layout.outputDense.1,layout.outputDense.2.1,layout.outputDense.2.2,Nat.mul_one]
    apply fits_address _ _ _ storage.output
    have hi := linear_bound _ _ _ _ hb hh
    have all := linear_bound _ _ _ _ hi hd
    nlinarith

/-- Source block/lane indexing coincides with the logical active addresses;
this includes the repaired probability block stride. -/
theorem source_addresses {α : Type} [Inhabited α] (c : Call) (m : Memory α)
    (b h start lane d : Nat) :
    probOffset (params c) (state m (b,h)) lane+start*(params c).probTokenStride=
      h*c.probability.rowStride+(startAt c m b+(start+lane))*c.probability.colStride ∧
    gatherOffset (params c) (state m (b,h)) lane+start*(params c).requestTokenStride=
      requestAt c m b*c.pages.rowStride+(windowAt c m b+start+lane)*c.pages.colStride ∧
    valueOffset (params c) (state m (b,h)) d=
      (h/group c)*c.value.headStride+d*c.value.dimStride := by
  simp [probOffset,gatherOffset,valueOffset,startIndex,requestIndex,params,state,startAt,requestAt,sequenceLength,windowStart,fullLength,windowAt]
  constructor <;> ring

#axiomsClean accesses
#axiomsClean source_addresses
end
end VeriTile.Bench.Repairs.TokenAttnMistral.Host
