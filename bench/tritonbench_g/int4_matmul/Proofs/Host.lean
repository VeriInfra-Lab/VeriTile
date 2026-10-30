import bench.tritonbench_g.int4_matmul.Proofs.Physical
import bench.tritonbench_g.int4_matmul.Proofs.Helpers
namespace VeriTile.Bench.Repairs.Int4Matmul.Host
set_option maxHeartbeats 6000000

def sourceTiles : List Tile := [
⟨64,128,32,8,1⟩,
⟨128,32,32,8,1⟩,
⟨64,64,32,8,1⟩,
⟨64,128,32,8,1⟩,
⟨64,64,64,8,1⟩,
⟨32,32,128,8,1⟩,
⟨32,32,256,8,1⟩,
⟨32,32,256,16,1⟩,
⟨32,32,512,16,1⟩,
⟨64,128,32,8,2⟩,
⟨128,32,32,8,2⟩,
⟨64,64,32,8,2⟩,
⟨64,128,32,8,2⟩,
⟨64,64,64,8,2⟩,
⟨32,32,128,8,2⟩,
⟨32,32,256,8,2⟩,
⟨32,32,256,16,2⟩,
⟨32,32,512,16,2⟩,
⟨64,16,32,8,1⟩,
⟨128,16,32,8,1⟩,
⟨64,16,32,8,1⟩,
⟨64,16,32,8,1⟩,
⟨64,16,64,8,1⟩,
⟨32,16,128,8,1⟩,
⟨32,16,256,8,1⟩,
⟨32,16,256,16,1⟩,
⟨32,16,512,16,1⟩,
⟨64,16,32,8,2⟩,
⟨128,16,32,8,2⟩,
⟨64,16,32,8,2⟩,
⟨64,16,32,8,2⟩,
⟨64,16,64,8,2⟩,
⟨32,16,128,8,2⟩,
⟨32,16,256,8,2⟩,
⟨32,16,256,16,2⟩,
⟨32,16,512,16,2⟩]

theorem all_tiles_valid (t : Tile) (h : t∈sourceTiles) : t.Valid := by
  simp only [sourceTiles,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h
  all_goals subst t;constructor <;> decide +kernel

theorem all_tiles_divide1024 (t : Tile) (h : t∈sourceTiles) : t.bk*t.splits∣1024 := by
  simp only [sourceTiles,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h
  all_goals subst t;decide +kernel

def kernelKGuard (s : Shape) := ceil s.k 1024*1024+1024

theorem ceil_le_multiple (k b upper : Nat) (hb : 0<b) (hk : k≤upper) (hu : b∣upper) :
    ceil k b*b≤upper := by
  obtain ⟨q,hq⟩ := hu
  have hd : ceil k b≤q := by
    unfold ceil
    apply (Nat.div_le_iff_le_mul hb).mpr
    have hqb : k≤q*b := by simpa [hq,Nat.mul_comm] using hk
    omega
  rw [hq]
  nlinarith

theorem tile_k_guard (s : Shape) (t : Tile) (member : t∈sourceTiles) :
    kEnvelope s t≤kernelKGuard s := by
  have ht := all_tiles_valid t member
  have hp := Nat.mul_pos ht.bkpos (split_positive t ht)
  have hd := all_tiles_divide1024 t member
  have hsize : t.bk*t.splits≤1024 := Nat.le_of_dvd (by decide) hd
  have cover := ceil_cover s.k 1024 (by decide)
  have hdiv : t.bk*t.splits∣ceil s.k 1024*1024 := dvd_mul_of_dvd_right hd _
  have h := ceil_le_multiple s.k (t.bk*t.splits) (ceil s.k 1024*1024) hp cover hdiv
  unfold kEnvelope rounds kernelKGuard
  nlinarith

theorem all_tiles_dim_divide (t : Tile) (h : t∈sourceTiles) : t.bm∣128 ∧ t.bn∣128 := by
  simp only [sourceTiles,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h
  all_goals subst t;constructor <;> decide +kernel

structure Guard (s : Shape) : Prop where
  a : s.m*s.k+kernelKGuard s<2^31
  q : (kernelKGuard s/8+1)*s.n<2^31
  scale : (kernelKGuard s/s.group+1)*s.n<2^31
  zero : (kernelKGuard s/s.group+1)*(s.n/8)<2^31
  output : ceil s.m 128*128*s.n+ceil s.n 128*128<2^31
  mEnvelope : ceil s.m 128*128<2^31
  nEnvelope : ceil s.n 128*128<2^31
  kEnvelope : kernelKGuard s<2^31
  gridEnvelope : (ceil s.m 128*128)*(ceil s.n 128*128)<2^31
  helper : s.k*s.n<2^31

theorem guarded_indices (s : Shape) (guard : Guard s) (t : Tile) (member : t∈sourceTiles) : IndexDomain s t := by
  have ht := all_tiles_valid t member
  have hk := tile_k_guard s t member
  have hq := Nat.div_le_div_right (c:=8) hk
  have hg := Nat.div_le_div_right (c:=s.group) hk
  obtain ⟨hdm,hdn⟩ := all_tiles_dim_divide t member
  have hm := ceil_le_multiple s.m t.bm (ceil s.m 128*128) ht.bmpos
    (ceil_cover _ _ (by decide)) (dvd_mul_of_dvd_right hdm _)
  have hn := ceil_le_multiple s.n t.bn (ceil s.n 128*128) ht.bnpos
    (ceil_cover _ _ (by decide)) (dvd_mul_of_dvd_right hdn _)
  refine ⟨?_,?_,?_,?_,?_⟩
  · exact lt_of_le_of_lt (Nat.add_le_add_left hk _) guard.a
  · exact lt_of_le_of_lt (Nat.mul_le_mul_right s.n (Nat.add_le_add_right hq 1)) guard.q
  · exact lt_of_le_of_lt (Nat.mul_le_mul_right s.n (Nat.add_le_add_right hg 1)) guard.scale
  · exact lt_of_le_of_lt (Nat.mul_le_mul_right (s.n/8) (Nat.add_le_add_right hg 1)) guard.zero
  · exact lt_of_le_of_lt (Nat.add_le_add (Nat.mul_le_mul_right s.n hm) hn) guard.output

theorem all_tiles_group_le_n (t : Tile) (h : t∈sourceTiles) : t.gm≤t.bn := by
  simp only [sourceTiles,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h
  all_goals subst t;decide +kernel

theorem grid_indices (s : Shape) (guard : Guard s) (t : Tile) (member : t∈sourceTiles) :
    blocksM s t*blocksN s t<2^31 ∧ t.gm*blocksN s t<2^31 := by
  have ht := all_tiles_valid t member
  obtain ⟨hdm,hdn⟩ := all_tiles_dim_divide t member
  have hm := ceil_le_multiple s.m t.bm (ceil s.m 128*128) ht.bmpos
    (ceil_cover _ _ (by decide)) (dvd_mul_of_dvd_right hdm _)
  have hn := ceil_le_multiple s.n t.bn (ceil s.n 128*128) ht.bnpos
    (ceil_cover _ _ (by decide)) (dvd_mul_of_dvd_right hdn _)
  have hmb : blocksM s t≤ceil s.m 128*128 := by unfold blocksM;nlinarith [ht.bmpos]
  have hnb : blocksN s t≤ceil s.n 128*128 := by unfold blocksN;nlinarith [ht.bnpos]
  refine ⟨lt_of_le_of_lt (Nat.mul_le_mul hmb hnb) guard.gridEnvelope,?_⟩
  have hg := all_tiles_group_le_n t member
  have hn' : blocksN s t*t.bn≤ceil s.n 128*128 := hn
  exact lt_of_le_of_lt (by nlinarith) guard.nEnvelope

#axiomsClean grid_indices

inductive DType where
  | fp32 | int32 | fp16
  deriving DecidableEq
structure Tensor where
  allocation : Nat
  rows : Nat
  columns : Nat
  rowStride : Nat
  columnStride : Nat
  dtype : DType
  device : Nat
  offset : Nat
  lazyNeg : Bool
  lazyConj : Bool
  requiresGrad : Bool
  deriving DecidableEq

def Tensor.Canonical (x : Tensor) (dtype : DType) (rows columns device : Nat) :=
  x.rows=rows ∧ x.columns=columns ∧ x.rowStride=columns ∧ x.columnStride=1 ∧
  x.dtype=dtype ∧ x.device=device ∧ x.offset=0 ∧ x.lazyNeg=Bool.false ∧ x.lazyConj=Bool.false ∧ x.requiresGrad=Bool.false

/-- A canonical [columns,rows] allocation is transposed, then contiguous() copies
only if necessary. Singleton dimensions retain the transpose's actual strides. -/
def transposeRowStride (rows columns : Nat) : Nat :=
  if rows=1 ∨ columns=1 then 1 else columns
def transposeColumnStride (rows columns : Nat) : Nat :=
  if rows=1 ∨ columns=1 then rows else 1
def Tensor.TransposeContiguous (x : Tensor) (dtype : DType) (rows columns device : Nat) :=
  x.rows=rows ∧ x.columns=columns ∧ x.rowStride=transposeRowStride rows columns ∧
  x.columnStride=transposeColumnStride rows columns ∧ x.dtype=dtype ∧ x.device=device ∧
  x.offset=0 ∧ x.lazyNeg=Bool.false ∧ x.lazyConj=Bool.false ∧ x.requiresGrad=Bool.false

/-- Exactly the original canonical ABI or the quantizer's returned layout.
This does not admit arbitrary strides on singleton axes. -/
def Tensor.PackedLayout (x : Tensor) (dtype : DType) (rows columns device : Nat) :=
  x.Canonical dtype rows columns device ∨ x.TransposeContiguous dtype rows columns device

def Tensor.elementOffset (x : Tensor) (row column : Nat) :=
  row*x.rowStride+column*x.columnStride

theorem Tensor.packed_active_offset (x : Tensor) (dtype : DType) (rows columns device r c : Nat)
    (h : x.PackedLayout dtype rows columns device) (hr : r<rows) (hc : c<columns) :
    x.elementOffset r c=r*columns+c := by
  rcases h with h|h
  · simp [Tensor.elementOffset,h.2.2.1,h.2.2.2.1]
  · simp only [Tensor.elementOffset,h.2.2.1,h.2.2.2.1,transposeRowStride,transposeColumnStride]
    split_ifs with singleton
    · rcases singleton with one|one
      · have : r=0 := by omega
        simp [one,this]
      · have : c=0 := by omega
        simp [one,this]
    · simp

/-- Even masked/final row coordinates use a stride no larger than canonical;
the noncanonical column stride occurs only when the column coordinate is zero. -/
theorem Tensor.packed_formed_offset_le (x : Tensor) (dtype : DType) (rows columns device r c : Nat)
    (h : x.PackedLayout dtype rows columns device) (hc : c<columns) :
    x.elementOffset r c≤r*columns+c := by
  rcases h with h|h
  · simp [Tensor.elementOffset,h.2.2.1,h.2.2.2.1]
  · simp only [Tensor.elementOffset,h.2.2.1,h.2.2.2.1,transposeRowStride,transposeColumnStride]
    split_ifs with singleton
    · rcases singleton with one|one
      · simp only [one,Nat.mul_one]
        have : 1≤columns := by omega
        nlinarith
      · have : c=0 := by omega
        simp [one,this]
    · simp

#axiomsClean Tensor.packed_active_offset
#axiomsClean Tensor.packed_formed_offset_le

structure Arguments where
  shape : Shape
  input : Tensor
  packed : Tensor
  scale : Tensor
  zeros : Tensor
  supplied : Option Tensor

/-- Ordinary Tensor and fresh-allocation boundary; no memory/result theorem appears here. -/
structure Allocation where
  output : Tensor
  canonical : output.Canonical .fp32 output.rows output.columns output.device
  fresh : Nat

structure Contract (x : Arguments) : Prop where
  shape : x.shape.Valid
  input : x.input.Canonical .fp32 x.shape.m x.shape.k x.input.device
  packed : x.packed.PackedLayout .int32 (x.shape.k/8) x.shape.n x.input.device
  scale : x.scale.PackedLayout .fp32 (x.shape.k/x.shape.group) x.shape.n x.input.device
  zeros : x.zeros.PackedLayout .int32 (x.shape.k/x.shape.group) (x.shape.n/8) x.input.device
  independent : [x.input.allocation,x.packed.allocation,x.scale.allocation,x.zeros.allocation].Nodup
  supplied : ∀ out,x.supplied=some out → out.Canonical .fp32 x.shape.m x.shape.n x.input.device ∧
    out.allocation∉[x.input.allocation,x.packed.allocation,x.scale.allocation,x.zeros.allocation]
  indices : Guard x.shape
  packedAdvance : 1024*x.packed.rowStride<2^31

/-- Bridges the actual stride arguments passed to Triton to the existing dense
semantic memory, including masked coordinates and the final pointer advance. -/
structure StrideBinding (x : Arguments) (t : Tile) : Prop where
  active : ∀ k c,k<x.shape.k → c<x.shape.n →
    x.packed.elementOffset (k/8) c=qAddress x.shape k c ∧
    x.scale.elementOffset (k/x.shape.group) c=sAddress x.shape k c ∧
    x.zeros.elementOffset (k/x.shape.group) (c/8)=zAddress x.shape k c
  formed : ∀ k c,k<kEnvelope x.shape t → c<x.shape.n →
    x.packed.elementOffset (k/8) c<2^31 ∧
    x.scale.elementOffset (k/x.shape.group) c<2^31 ∧
    x.zeros.elementOffset (k/x.shape.group) (c/8)<2^31
  rawAdvance : t.bk*t.splits*x.packed.rowStride<2^31
  advance : ∀ sp block lane,
    ((sp*t.bk+lane)/8)*x.packed.rowStride+block*(t.bk*t.splits*x.packed.rowStride/8)=
      (kIndex t sp block lane/8)*x.packed.rowStride

theorem contract_strides (x : Arguments) (valid : Contract x) (t : Tile) (member : t∈sourceTiles) :
    StrideBinding x t := by
  have hs := valid.shape
  have ht := all_tiles_valid t member
  have domain := guarded_indices x.shape valid.indices t member
  refine ⟨?_,?_,?_,fun sp block lane => packed_advance t ht sp block lane x.packed.rowStride⟩
  · intro k c hk hc
    have hk8 : k/8<x.shape.k/8 := (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
      (by simpa [Nat.div_mul_cancel hs.k8] using hk)
    have hkg : k/x.shape.group<x.shape.k/x.shape.group := (Nat.div_lt_iff_lt_mul hs.gpos).mpr
      (by simpa [Nat.div_mul_cancel hs.groups] using hk)
    have hc8 : c/8<x.shape.n/8 := (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
      (by simpa [Nat.div_mul_cancel hs.n8] using hc)
    exact ⟨x.packed.packed_active_offset _ _ _ _ _ _ valid.packed hk8 hc,
      x.scale.packed_active_offset _ _ _ _ _ _ valid.scale hkg hc,
      x.zeros.packed_active_offset _ _ _ _ _ _ valid.zeros hkg hc8⟩
  · intro k c hk hc
    have hc8 : c/8<x.shape.n/8 := (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
      (by simpa [Nat.div_mul_cancel hs.n8] using hc)
    have bounds := formed_addresses x.shape t hs ht domain 0 c k hs.mpos hc hk
    exact ⟨lt_of_le_of_lt (x.packed.packed_formed_offset_le _ _ _ _ _ _ valid.packed hc) bounds.2.1,
      lt_of_le_of_lt (x.scale.packed_formed_offset_le _ _ _ _ _ _ valid.scale hc) bounds.2.2.1,
      lt_of_le_of_lt (x.zeros.packed_formed_offset_le _ _ _ _ _ _ valid.zeros hc8) bounds.2.2.2⟩
  · have size : t.bk*t.splits≤1024 := Nat.le_of_dvd (by decide) (all_tiles_divide1024 t member)
    exact lt_of_le_of_lt (Nat.mul_le_mul_right x.packed.rowStride size) valid.packedAdvance

#axiomsClean contract_strides

/-- Ordinary successful fresh allocation, disjoint from all input allocations. -/
structure FreshOutput (x : Arguments) (fresh : Tensor) : Prop where
  layout : fresh.Canonical .fp32 x.shape.m x.shape.n x.input.device
  independent : fresh.allocation∉[x.input.allocation,x.packed.allocation,x.scale.allocation,x.zeros.allocation]

structure Return (α : Type) where
  tensor : Tensor
  state : State α
  callerDevice : Nat

def outputTensor (x : Arguments) (fresh : Tensor) := x.supplied.getD fresh

def hostCall (x : Arguments) (t : Tile) (p : Ops α) (ws : List Work) (fresh : Tensor)
    (callerDevice : Nat) (initial : State α) : Return α :=
  ⟨outputTensor x fresh,call x.shape t p ws initial,callerDevice⟩

theorem supplied_identity (x : Arguments) (out fresh : Tensor) (h : x.supplied=some out) :
    outputTensor x fresh=out := by simp [outputTensor,h]

theorem fresh_identity (x : Arguments) (fresh : Tensor) (h : x.supplied=none) :
    outputTensor x fresh=fresh := by simp [outputTensor,h]

theorem return_geometry (x : Arguments) (valid : Contract x) (fresh : Tensor)
    (h : fresh.Canonical .fp32 x.shape.m x.shape.n x.input.device) :
    (outputTensor x fresh).Canonical .fp32 x.shape.m x.shape.n x.input.device := by
  cases ho : x.supplied with
  | none => simpa [outputTensor,ho] using h
  | some out => simpa [outputTensor,ho] using (valid.supplied out ho).1

theorem caller_device_restored (x : Arguments) (t : Tile) (p : Ops α) (ws : List Work)
    (fresh : Tensor) (device : Nat) (initial : State α) :
    (hostCall x t p ws fresh device initial).callerDevice=device := rfl

#axiomsClean all_tiles_valid
#axiomsClean all_tiles_divide1024
#axiomsClean tile_k_guard
#axiomsClean guarded_indices
#axiomsClean supplied_identity
#axiomsClean return_geometry
#axiomsClean caller_device_restored
end VeriTile.Bench.Repairs.Int4Matmul.Host
