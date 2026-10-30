import bench.tritonbench_g.multinomial_sampling.Proofs.KernelSource
import bench.tritonbench_g.multinomial_sampling.Proofs.Core.FPReference
import bench.tritonbench_g.multinomial_sampling.Contract
import bench.tritonbench_g.multinomial_sampling.Proofs.Spec
import VeriTile.Torch.FP32
import VeriTile.Triton.Memory.Bounds

namespace VeriTile.Bench.Repairs.MultinomialSampling.KernelMemory
open VeriTile.Triton FP.Structural SourceValue Contract
noncomputable section
variable {α : Type}

/-- The source's fixed launch configuration and exactly the padded index
checks executed in the kernel branch of the Python wrapper. -/
structure Domain (p : Parameters) (scoreBase labelBase seedBase offsetBase outputBase : Nat) : Prop where
  block : p.block = 8
  blockTokens : p.blockTokens = 128
  tokens : 1 < p.tokens
  shape : shapeCheck p.batches p.tokens = Bool.true
  rows : paddedRows p.batches < 2^31
  columns : paddedTokens p.tokens < 2^31
  scores : matrixWindow (paddedRows p.batches) (paddedTokens p.tokens)
    p.strideScoreBatch p.strideScoreToken scoreBase 4 = Bool.true
  labels : matrixWindow (paddedRows p.batches) (paddedTokens p.tokens)
    p.strideIndexBatch p.strideIndexToken labelBase 8 = Bool.true
  seeds : vectorWindow (paddedRows p.batches) seedBase = Bool.true
  offsets : vectorWindow (paddedRows p.batches) offsetBase = Bool.true
  output : vectorWindow (paddedRows p.batches) outputBase = Bool.true

def grid (p : Parameters) : Nat := (p.batches+7)/8

theorem row_bound (p : Parameters) (hb : p.block=8) (s : State α)
    (launch : s.pids 0 < grid p) (i : Row p) : row p s i < paddedRows p.batches := by
  have hi : i.1.val<8 := by simpa only [hb] using i.1.isLt
  have hl : s.pids 0<(p.batches+7)/8 := launch
  simp only [row,hb,paddedRows]
  omega

/-- Exactly one launched program and local row writes each requested batch. -/
theorem output_coverage (p : Parameters) (hb : p.block=8) (r : Nat) (hr : r<p.batches) :
    ∃! location : Fin (grid p) × Fin p.block,location.1.val*p.block+location.2.val=r := by
  have coverage : p.batches≤grid p*8 := by dsimp [grid]; omega
  have hdiv : r/8<grid p := by omega
  have hmod : r%8<p.block := by rw [hb]; omega
  refine ⟨(⟨r/8,hdiv⟩,⟨r%8,hmod⟩),?_,?_⟩
  · simp only [hb]
    omega
  · intro other he
    have ho : other.2.val<8 := by simpa only [hb] using other.2.isLt
    simp only [hb] at he
    apply Prod.ext
    · apply Fin.ext
      dsimp only
      omega
    · apply Fin.ext
      dsimp only
      omega

theorem token_bound (p : Parameters) (hb : p.blockTokens=128)
    (k : Nat) (hk : k<count p) (j : Fin p.blockTokens) :
    k*p.blockTokens+j.val < paddedTokens p.tokens := by
  have hj : j.val<128 := by simpa only [hb] using j.isLt
  have hcount : count p=(p.tokens+127)/128 := by simp [count,hb]
  rw [hcount] at hk
  simp only [hb,paddedTokens]
  omega

theorem matrix_index {rows tokens sb st base width r t : Nat}
    (h : matrixWindow rows tokens sb st base width = Bool.true)
    (hr : r < rows) (ht : t < tokens) :
    r*sb+t*st < 2^31 ∧ base+width*(r*sb+t*st) < 2^64 := by
  simp only [matrixWindow,decide_eq_true_eq] at h
  have hm : r*sb+t*st ≤ last rows tokens sb st := by
    unfold last
    exact Nat.add_le_add (Nat.mul_le_mul_right sb (by omega)) (Nat.mul_le_mul_right st (by omega))
  exact ⟨lt_of_le_of_lt hm h.2.2.1,
    lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left width hm) base) h.2.2.2.2⟩

theorem vector_index {rows base r : Nat}
    (h : vectorWindow rows base = Bool.true) (hr : r < rows) : base+8*r < 2^64 := by
  simp only [vectorWindow,decide_eq_true_eq] at h
  omega

/-- Every eager matrix/vector pointer is finite, including masked lanes.
These are integer/address facts and use no floating-point conditions. -/
theorem formed_addresses (p : Parameters) (sb ib seb ob yb : Nat)
    (h : Domain p sb ib seb ob yb) (s : State α)
    (launch : s.pids 0 < grid p) (i : Row p) (k : Nat) (hk : k<count p)
    (j : Fin p.blockTokens) :
    let r := row p s i
    let t := k*p.blockTokens+j.val
    r<2^31 ∧ t<2^31 ∧
    r*p.strideScoreBatch+t*p.strideScoreToken<2^31 ∧
    r*p.strideIndexBatch+t*p.strideIndexToken<2^31 ∧
    sb+4*(r*p.strideScoreBatch+t*p.strideScoreToken)<2^64 ∧
    ib+8*(r*p.strideIndexBatch+t*p.strideIndexToken)<2^64 ∧
    seb+8*r<2^64 ∧ ob+8*r<2^64 ∧ yb+8*r<2^64 := by
  have hr := row_bound p h.block s launch i
  have ht := token_bound p h.blockTokens k hk j
  have hs := matrix_index h.scores hr ht
  have hi := matrix_index h.labels hr ht
  exact ⟨hr.trans h.rows,ht.trans h.columns,hs.1,hi.1,hs.2,hi.2,
    vector_index h.seeds hr,vector_index h.offsets hr,vector_index h.output hr⟩

theorem selected_address (p : Parameters) (sb ib seb ob yb : Nat)
    (h : Domain p sb ib seb ob yb) (s : State α) (launch : s.pids 0<grid p)
    (i : Row p) (position : Nat) :
    let address := row p s i*p.strideIndexBatch+min position (p.tokens-1)*p.strideIndexToken
    address<2^31 ∧ ib+8*address<2^64 := by
  have hr := row_bound p h.block s launch i
  have cover : p.tokens≤paddedTokens p.tokens := by dsimp [paddedTokens]; omega
  have bound : min position (p.tokens-1)<paddedTokens p.tokens := by
    have := Nat.min_le_right position (p.tokens-1)
    have := h.tokens
    omega
  exact matrix_index h.labels hr bound

theorem signed32_exact (n : Nat) (hn : n<2^31) : (BitVec.ofNat 32 n).toInt=(n : Int) := by
  have hnat : (BitVec.ofNat 32 n).toNat=n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hnat];omega),hnat]

/-- Native signed32 products and sums denote the proved nonnegative address. -/
theorem matrix_native_exact (r t sb st : Nat) (h : r*sb+t*st<2^31) :
    (BitVec.ofNat 32 r*BitVec.ofNat 32 sb+
      BitVec.ofNat 32 t*BitVec.ofNat 32 st).toInt=((r*sb+t*st : Nat) : Int) := by
  simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  exact signed32_exact _ h

theorem clamp_active (tokens position : Nat) (positive : 0<tokens) :
    min position (tokens-1)<tokens := by
  have := Nat.min_le_right position (tokens-1)
  omega

/-- Logical region capacities for active scalar loads/stores. These windows
are supplied by the ordinary Tensor storage binding, independently of values. -/
structure Windows (p : Parameters) (capacity : RegionBounds) : Prop where
  scores : ∀ r j, r<p.batches → j<p.tokens →
    r*p.strideScoreBatch+j*p.strideScoreToken<capacity p.scores
  labels : ∀ r j, r<p.batches → j<p.tokens →
    r*p.strideIndexBatch+j*p.strideIndexToken<capacity p.indices
  seeds : p.batches≤capacity p.seeds
  offsets : p.batches≤capacity p.offsets
  output : p.batches≤capacity p.output

theorem selected_capacity (p : Parameters) (capacity : RegionBounds)
    (h : Windows p capacity) (positive : 0<p.tokens) (r position : Nat) (hr : r<p.batches) :
    r*p.strideIndexBatch+min position (p.tokens-1)*p.strideIndexToken<capacity p.indices :=
  h.labels r _ hr (clamp_active p.tokens position positive)

/-- Same-call Torch score metadata and stride observations. The stride law is
an ordinary indexing API relation; it contains no kernel-result assumption. -/
structure Scores (p : Parameters) (t : VeriTile.Torch.FP32.Tensor) : Prop where
  laws : VeriTile.Torch.FP32.TensorLaws t
  shape : t.metadata.shape=[p.batches,p.tokens]
  contiguous : t.metadata.isContiguous=Bool.true
  ordinary : t.metadata.isNeg=Bool.false
  aligned : 0<t.metadata.numel → t.dataPtr.toNat%4=0
  strides : ∀ r j, r<p.batches → j<p.tokens →
    t.physicalIndex (r*p.tokens+j)=t.storageOffset+r*p.strideScoreBatch+j*p.strideScoreToken

theorem flat_bound (batches tokens r j : Nat) (hr : r<batches) (hj : j<tokens) :
    r*tokens+j<batches*tokens := by nlinarith

theorem scores_logical (p : Parameters) (t : VeriTile.Torch.FP32.Tensor) (h : Scores p t)
    (read : Nat → VeriTile.Torch.FP32.Word) (r j : Nat) (hr : r<p.batches) (hj : j<p.tokens) :
    let v := VeriTile.Torch.FP32.dataView t h.laws h.contiguous
    r*p.strideScoreBatch+j*p.strideScoreToken<v.elements ∧
    read (v.address (r*p.strideScoreBatch+j*p.strideScoreToken))=
      VeriTile.Torch.FP32.logicalValue read t h.laws (r*p.tokens+j) := by
  have hb : r*p.tokens+j<t.metadata.numel := by
    simpa [VeriTile.Torch.TensorMetadata.numel,h.shape] using flat_bound p.batches p.tokens r j hr hj
  have hc := h.laws.contiguous h.contiguous _ hb
  rw [h.strides r j hr hj] at hc
  have eq : r*p.strideScoreBatch+j*p.strideScoreToken=r*p.tokens+j := by omega
  dsimp only
  rw [eq]
  exact ⟨hb,VeriTile.Torch.FP32.direct_read_eq_logical read t h.laws h.contiguous h.ordinary _ hb⟩

theorem scores_pointer (p : Parameters) (t : VeriTile.Torch.FP32.Tensor) (h : Scores p t) :
    let v := VeriTile.Torch.FP32.dataView t h.laws h.contiguous
    v.base%4=0 ∧ (0<t.metadata.numel → BitVec.ofNat 64 (v.address 0)=t.dataPtr) :=
  ⟨VeriTile.Torch.FP32.dataView_aligned t h.laws h.contiguous h.aligned,
    VeriTile.Torch.FP32.actual_pointer t h.laws h.contiguous⟩

/-- Task-local signed64 Tensor view. It covers supplied contiguous labels and
expanded default labels (zero batch stride), plus the seed/offset vectors.
`physicalIndex` is the Tensor's logical-to-storage map. -/
structure IntegerTensor where
  metadata : VeriTile.Torch.TensorMetadata
  storageBase : Nat
  storageElements : Nat
  storageOffset : Nat
  rows : Nat
  columns : Nat
  strideBatch : Nat
  strideToken : Nat
  physicalIndex : Nat → Nat → Nat
  span : storageBase+8*storageElements≤2^64

structure IntegerLaws (t : IntegerTensor) : Prop where
  dtype : t.metadata.dtype=.other "int64"
  layout : t.metadata.layout=.strided
  extent : t.metadata.numel=t.rows*t.columns
  pointer : 0<t.rows → 0<t.columns →
    t.metadata.dataPtr.toNat=t.storageBase+8*t.storageOffset
  strides : ∀ r j, r<t.rows → j<t.columns →
    t.physicalIndex r j=t.storageOffset+r*t.strideBatch+j*t.strideToken
  capacity : ∀ r j, r<t.rows → j<t.columns → t.physicalIndex r j<t.storageElements

def integerLogical (read : Nat → BitVec 64) (t : IntegerTensor) (r j : Nat) : BitVec 64 :=
  let word := read (t.storageBase+8*t.physicalIndex r j)
  if t.metadata.isNeg then BitVec.ofInt 64 (-word.toInt) else word

theorem integer_logical (read : Nat → BitVec 64) (t : IntegerTensor) (h : IntegerLaws t)
    (ordinary : t.metadata.isNeg=Bool.false) (r j : Nat) (hr : r<t.rows) (hj : j<t.columns) :
    let address := t.metadata.dataPtr.toNat+8*(r*t.strideBatch+j*t.strideToken)
    read address=integerLogical read t r j ∧ address+8≤t.storageBase+8*t.storageElements := by
  have hp := h.pointer (by omega) (by omega)
  have hs := h.strides r j hr hj
  have hb := h.capacity r j hr hj
  dsimp only
  rw [hp]
  constructor
  · simp only [integerLogical,ordinary,Bool.false_eq_true,↓reduceIte,hs]
    congr 1
    omega
  · rw [hs] at hb
    omega

theorem signed64 (w : BitVec 64) : Random.IntegerType.i64.inRange w.toInt=Bool.true := by
  simp only [Random.IntegerType.inRange,decide_eq_true_eq]
  have lo := BitVec.le_toInt w
  have hi := BitVec.toInt_lt (x:=w)
  norm_num at lo hi ⊢
  omega

variable [Inhabited α]

theorem source_score (p : Parameters) (s : State α) (t : VeriTile.Torch.FP32.Tensor)
    (h : Scores p t) (read : Nat → VeriTile.Torch.FP32.Word) (denote : VeriTile.Torch.FP32.Word → α)
    (binding : ∀ a,a<t.metadata.numel → (s.mem p.scores a).read .real=
      denote (read ((VeriTile.Torch.FP32.dataView t h.laws h.contiguous).address a)))
    (r j : Nat) (hr : r<p.batches) (hj : j<p.tokens) :
    (s.mem p.scores (r*p.strideScoreBatch+j*p.strideScoreToken)).read .real=
      denote (VeriTile.Torch.FP32.logicalValue read t h.laws (r*p.tokens+j)) := by
  obtain ⟨capacity,logical⟩ := scores_logical p t h read r j hr hj
  exact (binding _ capacity).trans (congrArg denote logical)

theorem source_integer (s : State α) (region : RegionName) (read : Nat → BitVec 64)
    (t : IntegerTensor) (h : IntegerLaws t) (ordinary : t.metadata.isNeg=Bool.false)
    (binding : ∀ r j,r<t.rows → j<t.columns →
      (s.mem region (r*t.strideBatch+j*t.strideToken)).read .int=
        (read (t.metadata.dataPtr.toNat+8*(r*t.strideBatch+j*t.strideToken))).toInt)
    (r j : Nat) (hr : r<t.rows) (hj : j<t.columns) :
    (s.mem region (r*t.strideBatch+j*t.strideToken)).read .int=(integerLogical read t r j).toInt := by
  rw [binding r j hr hj,(integer_logical read t h ordinary r j hr hj).1]

/-- Active seeds and undefined masked seeds are ordinary signed64 payloads.
The typed undefined value is arbitrary; no fixed seed is assumed. -/
theorem seed_domain (p : Parameters) (s : State α)
    (loads : ∀ r, r<p.batches → ∃ w : BitVec 64,(s.mem p.seeds r).read .int=w.toInt)
    (undef : ∀ r,∃ w : BitVec 64,s.undef .int p.seeds r=w.toInt) : SeedDomain p s := by
  intro i
  unfold seedValue intLoad
  split
  · rename_i ha
    have hr : row p s i<p.batches := by simpa only [active,decide_eq_true_eq] using ha
    obtain ⟨w,hw⟩ := loads _ hr
    rw [hw]
    exact signed64 w
  · obtain ⟨w,hw⟩ := undef (row p s i)
    rw [hw]
    exact signed64 w

/-- Every active output remains a signed64 input label. The pointer clamp
makes this independent of all floating-point primitives and sample values. -/
theorem output_signed64 (P : Primitives α) (p : Parameters) (s : State α)
    (i : Row p) (ha : active p s i) (positive : 0<p.tokens)
    (labels : ∀ j,j<p.tokens → Random.IntegerType.i64.inRange
      ((s.mem p.indices (row p s i*p.strideIndexBatch+j*p.strideIndexToken)).read .int)=Bool.true)
    (k : Nat) : Random.IntegerType.i64.inRange ((rounds P .fp32 p s k).output i)=Bool.true := by
  induction k with
  | zero =>
    simpa [rounds,initialAccumulator,initialOutput,intLoad,ha] using labels 0 positive
  | succ k ih =>
    change Random.IntegerType.i64.inRange (if found P .fp32 p s (k*p.blockTokens)
      (rounds P .fp32 p s k) i then selectedIndex P .fp32 p s (k*p.blockTokens)
      (rounds P .fp32 p s k) i else (rounds P .fp32 p s k).output i)=Bool.true
    by_cases hf : found P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i=Bool.true
    · simp only [hf,↓reduceIte,selectedIndex,ha,decide_true,Bool.and_self,↓reduceIte]
      exact labels _ (clamp_active p.tokens _ positive)
    · simpa only [hf,↓reduceIte] using ih

/-- Exact integer-unit readback used by the executed Python row guard.
This derives the mathematical domain from input scalar observations. -/
theorem probability_domain (tokens : Nat) (positive : 0<tokens)
    (finite : Fin tokens → Bool) (units : Fin tokens → Int) (weights : Nat → ℝ)
    (checked : probabilityRow tokens finite units=Bool.true)
    (readback : ∀ j : Fin tokens,weights j.val=(units j : ℝ)/(2:ℝ)^149) :
    Spec.Domain tokens weights := by
  simp only [probabilityRow,decide_eq_true_eq] at checked
  refine ⟨positive,?_,?_⟩
  · intro j hj
    rw [readback ⟨j,hj⟩]
    exact div_nonneg (Int.cast_nonneg (checked.1 ⟨j,hj⟩).2) (by positivity)
  · unfold Spec.cumulative
    rw [← Fin.sum_univ_eq_sum_range]
    simp_rw [readback]
    rw [← Finset.sum_div,← Int.cast_sum,checked.2]
    norm_num

#axiomsClean output_coverage
#axiomsClean formed_addresses
#axiomsClean selected_address
#axiomsClean matrix_native_exact
#axiomsClean selected_capacity
#axiomsClean scores_logical
#axiomsClean scores_pointer
#axiomsClean integer_logical
#axiomsClean source_score
#axiomsClean source_integer
#axiomsClean seed_domain
#axiomsClean probability_domain
#axiomsClean output_signed64
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.KernelMemory
