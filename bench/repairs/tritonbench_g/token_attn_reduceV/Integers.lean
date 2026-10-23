import bench.repairs.tritonbench_g.token_attn_reduceV.Bounds
namespace VeriTile.Bench.Repairs.TokenAttnReduceV.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

/-- The four span inequalities are exactly the wrapper's padded signed-int32
checks. All quantities are tensor shapes/strides, never an execution oracle. -/
structure IntegerDomain (c : Call) : Prop where
  probability : matrixExtent (heads c) c.probability.cols c.probability.rowStride c.probability.colStride+
    128*c.probability.colStride<2^31
  table : c.pages.rows*c.pages.cols+c.probability.cols+128<2^31
  tokens : c.pages.cols+c.probability.cols+128<2^31
  value : c.value.slots*c.value.heads*c.value.width≤2^31-128
  output : batches c*heads c*c.value.width≤2^31-128
  metadata : ∀ t∈[c.request,c.start,c.length],
    vectorExtent t.size t.stride≤2^31-128

/-- Every formed loop index and padded P/page element offset is signed-int32.
The masks remain distinct; no P/V tail-validity equality is assumed. -/
structure Formed {α : Type} [Inhabited α] (c : Call) (m : Memory α) : Prop where
  loop : ∀ b<batches c, ∀ cur<seq c m b, ∀ lane<128,
    cur+128<2^31 ∧ cur+lane<2^31 ∧ (0 : Nat)+cur+lane<2^31
  probability : ∀ b<batches c, ∀ h<heads c, ∀ cur<seq c m b, ∀ lane<128,
    h*c.probability.rowStride+(startAt c m b+cur+lane)*c.probability.colStride<2^31
  table : ∀ b<batches c, ∀ cur<seq c m b, ∀ lane<128,
    requestAt c m b*c.pages.cols+((0 : Nat)+cur+lane)<2^31

theorem formed {α : Type} [Inhabited α] (c : Call) (m : Memory α)
    (data : Data c m) (domain : IntegerDomain c) : Formed c m := by
  constructor
  · intro b hb cur hc lane hl
    have hp := data.probabilityWindow b hb
    have hw : (0 : Nat)≤c.pages.cols := Nat.zero_le _
    have limit := domain.tokens
    omega
  · intro b hb h hh cur hc lane hl
    have hp := data.probabilityWindow b hb
    have rows : heads c≠0 := by omega
    have cols : c.probability.cols≠0 := by omega
    have hr := Nat.mul_le_mul_right c.probability.rowStride (show h≤heads c-1 by omega)
    have hk := Nat.mul_le_mul_right c.probability.colStride
      (show startAt c m b+cur+lane≤c.probability.cols-1+128 by omega)
    have lim := domain.probability
    simp only [matrixExtent,if_neg (show ¬(heads c=0 ∨ c.probability.cols=0) by omega)] at lim
    nlinarith
  · intro b hb cur hc lane hl
    have hp := data.probabilityWindow b hb
    have hw : (0 : Nat)≤c.pages.cols := Nat.zero_le _
    have hr := Nat.mul_le_mul_right c.pages.cols
      (show requestAt c m b+1≤c.pages.rows from data.requests b hb)
    have lim := domain.table
    nlinarith

/-- All active value/output element arithmetic fits signed int32 as well. -/
theorem active_tensor32 {α : Type} [Inhabited α] (c : Call) (m : Memory α)
    (layout : Layout c) (data : Data c m) (domain : IntegerDomain c)
    (b h n d : Nat) (hb : b<batches c) (hh : h<heads c) (hd : d<c.value.width)
    (hn : n+(0 : Nat)<seq c m b) :
    pageAt c m b n*c.value.slotStride+(h/group c)*c.value.headStride+d*c.value.dimStride<2^31 ∧
    b*c.output.slotStride+h*c.output.headStride+d*c.output.dimStride<2^31 := by
  simp only [layout.valueDense.1,layout.valueDense.2.1,layout.valueDense.2.2,
    layout.outputDense.1,layout.outputDense.2.1,layout.outputDense.2.2,Nat.mul_one]
  constructor
  · have hi := linear_bound _ _ _ _ (data.pages b hb n hn) (group_bound c layout h hh)
    have all := linear_bound _ _ _ _ hi hd
    have limit := domain.value
    norm_num at limit ⊢
    nlinarith
  · have hi := linear_bound _ _ _ _ hb hh
    have all := linear_bound _ _ _ _ hi hd
    have limit := domain.output
    norm_num at limit ⊢
    nlinarith

theorem active_metadata32 (c : Call) (domain : IntegerDomain c)
    (t : Vector) (mem : t∈[c.request,c.start,c.length])
    (b : Nat) (hb : b<t.size) : b*t.stride<2^31 := by
  have small := vector_bound t.size t.stride b hb
  have limit := domain.metadata t mem
  omega

/-- Each raw 32-bit word has the intended nonnegative signed interpretation. -/
theorem signed32 (n : Nat) (h : n<2^31) : (BitVec.ofNat 32 n).toInt=(n : Int) := by
  rw [BitVec.toInt_ofNat']
  apply Int.bmod_eq_of_le <;> omega

#axiomsClean active_tensor32
#axiomsClean active_metadata32
#axiomsClean formed
#axiomsClean signed32
end
end VeriTile.Bench.Repairs.TokenAttnReduceV.Host
