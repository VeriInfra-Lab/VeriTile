import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.HostCore
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
/-- Capacity counts four-byte elements, relative to the allocation base. -/
def Fits (t : Tensor) (extent : Nat) := t.storageOffset+extent ≤ t.capacity
def byteAddress (t : Tensor) (index : Nat) := t.allocationBase+4*(t.storageOffset+index)

/-- Exact element span, including zero-size views and zero/nonunit strides. -/
def vectorExtent (size stride : Nat) := if size=0 then 0 else (size-1)*stride+1
def matrixExtent (rows cols rowStride colStride : Nat) :=
  if rows=0 ∨ cols=0 then 0 else (rows-1)*rowStride+(cols-1)*colStride+1

structure Storage (c : Call) : Prop where
  input : Fits c.input.toTensor (matrixExtent (heads c) c.input.cols c.input.rowStride c.input.colStride)
  output : Fits c.output.toTensor (matrixExtent (heads c) c.input.cols c.output.rowStride c.output.colStride)
  start : Fits c.start.toTensor (vectorExtent c.start.size c.start.stride)
  length : Fits c.length.toTensor (vectorExtent c.length.size c.length.stride)
  physicalSeparate : ∀ t∈[c.input.toTensor,c.start.toTensor,c.length.toTensor],
    byteAddress c.output.toTensor (matrixExtent (heads c) c.input.cols c.output.rowStride c.output.colStride) ≤ byteAddress t 0 ∨
    t.allocationBase+4*t.capacity ≤ byteAddress c.output.toTensor 0

def InStorage (t : Tensor) (i : Nat) :=
  t.storageOffset+i < t.capacity ∧
  t.allocationBase ≤ byteAddress t i ∧
  byteAddress t i+4 ≤ t.allocationBase+4*t.capacity

theorem fits_address (t : Tensor) (n i : Nat) (fits : Fits t n) (hi : i < n) :
    InStorage t i := by
  simp only [Fits] at fits
  simp only [InStorage,byteAddress]
  omega

theorem linear_bound (a n b q : Nat) (ha : a < n) (hb : b < q) : a*q+b < n*q := by
  have step := Nat.mul_le_mul_right q (show a+1 ≤ n by omega)
  nlinarith

theorem vector_bound (size stride i : Nat) (hi : i < size) :
    i*stride < vectorExtent size stride := by
  have hne : size≠0 := by omega
  have h := Nat.mul_le_mul_right stride (show i ≤ size-1 by omega)
  simp only [vectorExtent,if_neg hne]
  omega

theorem matrix_bound (rows cols rs cs i j : Nat) (hi : i < rows) (hj : j < cols) :
    i*rs+j*cs < matrixExtent rows cols rs cs := by
  have hne : ¬(rows=0 ∨ cols=0) := by omega
  have hr := Nat.mul_le_mul_right rs (show i ≤ rows-1 by omega)
  have hc := Nat.mul_le_mul_right cs (show j ≤ cols-1 by omega)
  simp only [matrixExtent,if_neg hne]
  omega

/-- Exact active addresses and actual strided metadata addresses. There is
no requirement for masked padding pointers to lie within a tensor. -/
structure Accesses {α : Type} [Inhabited α] (c : Call) (m : Memory α) : Prop where
  metadata : ∀ b < batches c,
    InStorage c.start.toTensor (b*c.start.stride) ∧
    InStorage c.length.toTensor (b*c.length.stride)
  active : ∀ b < batches c, ∀ h < heads c, ∀ n < seq c m b,
    InStorage c.input.toTensor (h*c.input.rowStride+(startAt c m b+n)*c.input.colStride) ∧
    InStorage c.output.toTensor (h*c.output.rowStride+(startAt c m b+n)*c.output.colStride)

theorem accesses {α : Type} [Inhabited α] (c : Call) (m : Memory α)
    (layout : Layout c) (data : Data c m) (storage : Storage c) : Accesses c m := by
  constructor
  · intro b hb
    exact ⟨fits_address _ _ _ storage.start (vector_bound _ _ _ hb),
      fits_address _ _ _ storage.length (vector_bound _ _ _ (lt_of_lt_of_le hb layout.lengthShape))⟩
  · intro b hb h hh n hn
    have token := lt_of_lt_of_le (Nat.add_lt_add_left hn _) (data.window b hb)
    exact ⟨fits_address _ _ _ storage.input (matrix_bound _ _ _ _ _ _ hh token),
      fits_address _ _ _ storage.output (matrix_bound _ _ _ _ _ _ hh token)⟩


#axiomsClean accesses
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Host
