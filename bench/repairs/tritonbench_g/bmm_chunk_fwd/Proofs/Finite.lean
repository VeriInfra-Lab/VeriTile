import bench.repairs.tritonbench_g.bmm_chunk_fwd.Kernels
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.Geometry

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Finite
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd

/-- Successful ordinary allocations in the nonempty-launch branch. All input
storage choices use at least two bytes/element. Output overrides include
FP8, so the output bound uses only one byte/element. These are allocator bounds, not added user shape cutoffs. -/
structure AllocatedSizes (h : Host.Shape) : Prop where
  nonempty : 0<h.outputSize
  outputBytes : h.outputSize < 2^63
  inputBytes : 2*(h.batch*h.seqlen*h.groups*h.K) < 2^63

theorem positive_dimensions (h : Host.Shape) (s : AllocatedSizes h) :
    0<h.batch ∧ 0<h.chunks ∧ 0<h.groups ∧ 0<h.chunk := by
  have nz := s.nonempty
  refine ⟨?_,?_,?_,?_⟩ <;> by_contra hz
  · have : h.batch=0 := by omega
    simp [Host.Shape.outputSize,this] at nz
  · have : h.chunks=0 := by omega
    simp [Host.Shape.outputSize,this] at nz
  · have : h.groups=0 := by omega
    simp [Host.Shape.outputSize,this] at nz
  · have : h.chunk=0 := by omega
    simp [Host.Shape.outputSize,this] at nz

theorem dimensions_bounded (h : Host.Shape) (s : AllocatedSizes h) :
    h.seqlen<9223372036854775808 ∧ h.chunk<4294967296 ∧
    h.K<4611686018427387904 := by
  obtain ⟨hb,hch,hg,hc⟩ := positive_dimensions h s
  have ho : h.outputSize<9223372036854775808 := by have := s.outputBytes; norm_num at this; omega
  have hi : h.batch*h.seqlen*h.groups*h.K<4611686018427387904 := by
    have := s.inputBytes; norm_num at this; omega
  have pref : 0<h.batch*(h.chunks*h.groups) := Nat.mul_pos hb (Nat.mul_pos hch hg)
  have square : h.chunk*h.chunk≤h.outputSize := by
    simpa only [Host.Shape.outputSize, Nat.mul_assoc, Nat.one_mul] using
      Nat.mul_le_mul_right (h.chunk*h.chunk) (Nat.succ_le_of_lt pref)
  have chunk : h.chunk≤h.outputSize :=
    (Nat.le_mul_of_pos_right h.chunk hc).trans square
  have cover := Geometry.cdiv_covers (n:=h.seqlen) hc
  have chunked : h.chunks*h.chunk≤h.outputSize := by
    calc
      h.chunks*h.chunk ≤ h.chunks*h.chunk*(h.batch*h.groups*h.chunk) :=
        Nat.le_mul_of_pos_right _ (Nat.mul_pos (Nat.mul_pos hb hg) hc)
      _ = h.outputSize := by unfold Host.Shape.outputSize; ring
  have seq : h.seqlen≤h.outputSize := cover.trans chunked
  have hs : 0<h.seqlen := by
    by_contra zero
    have zeq : h.seqlen=0 := by omega
    have div : (h.chunk-1)/h.chunk=0 := Nat.div_eq_of_lt (by omega)
    simp only [Host.Shape.chunks,Host.cdiv,zeq,Nat.zero_add,div] at hch
    omega
  have kb : h.K≤h.batch*h.seqlen*h.groups*h.K :=
    Nat.le_mul_of_pos_left _ (Nat.mul_pos (Nat.mul_pos hb hs) hg)
  have hc32 : h.chunk<4294967296 := by nlinarith
  omega

/-- Exact integer ceilDiv, including host counts above binary64 precision. -/
theorem integer_chunks (seqlen chunk : Nat) :
    (seqlen+chunk-1)/chunk=Host.cdiv seqlen chunk := rfl

theorem block_base (n block pid : Nat) (positive : 0<block)
    (hp : pid<Host.cdiv n block) : pid*block<n := by
  have step : (pid+1)*block≤n+block-1 :=
    (Nat.le_div_iff_mul_le positive).mp (by simpa only [Host.cdiv] using (Nat.succ_le_of_lt hp))
  have sub : n+block-1+1=n+block := by omega
  nlinarith

theorem padded_index (n block pid lane : Nat) (positive : 0<block)
    (hp : pid<Host.cdiv n block) (hl : lane<block) : pid*block+lane<n+block := by
  have := block_base n block pid positive hp
  omega

theorem signed64 (n : Nat) (hn : n<9223372036854775808) :
    (BitVec.ofNat 64 n).toInt=n := by
  have hn' : (BitVec.ofNat 64 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]; exact Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt, hn']
  rw [hn']; norm_num; omega

theorem compare64 (a b : Nat) (ha : a<9223372036854775808) (hb : b<9223372036854775808) :
    (BitVec.ofNat 64 a).slt (BitVec.ofNat 64 b)=decide (a<b) := by
  simp only [BitVec.slt,signed64 a ha,signed64 b hb,decide_eq_decide]
  omega

/-- Every padded M/N lane and K lane stays representable before comparison.
This includes non-power-of-two chunks and partial K blocks. -/
theorem padded_masks (n block pid lane : Nat) (hn : n<4611686018427387904)
    (positive : 0<block) (small : block≤256) (hp : pid<Host.cdiv n block) (hl : lane<block) :
    (BitVec.ofNat 64 (pid*block+lane)).slt (BitVec.ofNat 64 n)=decide (pid*block+lane<n) := by
  apply compare64
  · have := padded_index n block pid lane positive hp hl; omega
  · omega

/-- Sum-of-products indexing is modular arithmetic. It may form an out-of-
allocation pointer on a masked lane; active offsets have exact native values
because the ordinary allocation bound is checked below. -/
def offset (batch row head column : Nat) (strides : Host.Strides) : BitVec 64 :=
  BitVec.ofNat 64 batch*BitVec.ofNat 64 strides.batch +
  BitVec.ofNat 64 row*BitVec.ofNat 64 strides.row +
  BitVec.ofNat 64 head*BitVec.ofNat 64 strides.head +
  BitVec.ofNat 64 column*BitVec.ofNat 64 strides.column

theorem offset_formula (batch row head column : Nat) (strides : Host.Strides) :
    offset batch row head column strides =
      BitVec.ofNat 64 (batch*strides.batch+row*strides.row+head*strides.head+column*strides.column) := by
  simp only [offset,←BitVec.ofNat_mul,←BitVec.ofNat_add]

/-- Address includes the data-pointer storage offset once. Byte widths are
2/4 for data and 4/8 for signed Seq. All accessed bytes, not just element
starts, lie within their allocation and the 64-bit address space. -/
theorem active_address (pointer bytes off capacity : Nat)
    (width : 0<bytes) (inside : bytes*(off+1)≤capacity) (fits : pointer+capacity≤2^64) :
    (BitVec.ofNat 64 pointer + BitVec.ofNat 64 bytes*BitVec.ofNat 64 off).toNat=pointer+bytes*off ∧
    pointer+bytes*off+bytes≤pointer+capacity := by
  have finish : pointer+bytes*off+bytes≤pointer+capacity := by nlinarith
  refine ⟨?_,finish⟩
  simp only [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  omega

theorem active_offset (b r g k : Nat) (strides : Host.Strides) (extent : Nat)
    (inside : b*strides.batch+r*strides.row+g*strides.head+k*strides.column<extent)
    (size : extent<2^63) :
    (offset b r g k strides).toNat=b*strides.batch+r*strides.row+g*strides.head+k*strides.column := by
  rw [offset_formula]
  simp only [BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  norm_num at *
  omega

theorem configuration_small (c : Host.Config) (hc : c∈Host.configurations) :
    c.M≤256 ∧ c.N≤256 ∧ c.K≤256 := by
  simp only [Host.configurations, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

def mask (pid block lane limit : Nat) : Bool :=
  (BitVec.ofNat 64 pid * BitVec.ofNat 64 block + BitVec.ofNat 64 lane).slt
    (BitVec.ofNat 64 limit)

theorem mask_exact (n block pid lane limit : Nat) (hn : n<4611686018427387904)
    (hb : 0<block) (hs : block≤256) (hp : pid<Host.cdiv n block)
    (hl : lane<block) (limit_le : limit≤n) :
    mask pid block lane limit=decide (pid*block+lane<limit) := by
  unfold mask
  rw [←BitVec.ofNat_mul, ←BitVec.ofNat_add]
  apply compare64
  · have := padded_index n block pid lane hb hp hl; omega
  · omega

/-- Native subtraction is exact at each reached K iteration or chunk base.
The masked comparison is equivalent to the original natural-index predicate. -/
theorem remainder_mask (n block pid lane : Nat) (hn : n<9223372036854775808)
    (hb : 0<block) (hs : block≤256) (hp : pid<Host.cdiv n block) (hl : lane<block) :
    (BitVec.ofNat 64 lane).slt
      (BitVec.ofNat 64 n - BitVec.ofNat 64 pid * BitVec.ofNat 64 block) =
      decide (pid*block+lane<n) := by
  have base := block_base n block pid hb hp
  rw [←BitVec.ofNat_mul, BitVec.ofNat_sub_ofNat_of_le n (pid*block) (by omega) (by omega)]
  rw [compare64 lane (n-pid*block) (by omega) (by omega)]
  simp only [decide_eq_decide]
  omega

def chunkLimit (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) :=
  min h.shape.chunk (h.shape.seqlen-Geometry.p2 p/h.shape.groups*h.shape.chunk)

/-- The certificate refers to the source's own PID quotient/remainder,
M/N aranges, last-chunk limit, causal boundary and every reached K iteration.
It is derived for all configurations, not assumed as a native-safe domain. -/
structure SourceIndices (h : Host.Call) (c : Host.Config) : Prop where
  chunkBase : ∀ p : Geometry.PID h c, Geometry.p2 p/h.shape.groups*h.shape.chunk<h.shape.seqlen
  causal : ∀ p : Geometry.PID h c,
    (BitVec.ofNat 64 (Geometry.p0 p % Host.cdiv h.shape.chunk c.N) * BitVec.ofNat 64 c.N).slt
      ((BitVec.ofNat 64 (Geometry.p0 p / Host.cdiv h.shape.chunk c.N) + 1) * BitVec.ofNat 64 c.M) =
      decide (Geometry.p0 p % Host.cdiv h.shape.chunk c.N*c.N <
        (Geometry.p0 p / Host.cdiv h.shape.chunk c.N+1)*c.M)
  rowMasks : ∀ (p : Geometry.PID h c) (i : Fin c.M) limit, limit≤h.shape.chunk →
    mask (Geometry.p0 p / Host.cdiv h.shape.chunk c.N) c.M i limit =
      decide (Geometry.p0 p / Host.cdiv h.shape.chunk c.N*c.M+i<limit)
  columnMasks : ∀ (p : Geometry.PID h c) (j : Fin c.N) limit, limit≤h.shape.chunk →
    mask (Geometry.p0 p % Host.cdiv h.shape.chunk c.N) c.N j limit =
      decide (Geometry.p0 p % Host.cdiv h.shape.chunk c.N*c.N+j<limit)
  kMasks : ∀ k, k<Host.cdiv h.shape.K c.K → ∀ i : Fin c.K,
    (BitVec.ofNat 64 i).slt (BitVec.ofNat 64 h.shape.K-
      BitVec.ofNat 64 k*BitVec.ofNat 64 c.K)=decide (k*c.K+i<h.shape.K)

theorem source_indices (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : AllocatedSizes h.shape) : SourceIndices h c := by
  obtain ⟨_,_,hg,hchunk⟩ := positive_dimensions h.shape sizes
  obtain ⟨hseq,hch,hk⟩ := dimensions_bounded h.shape sizes
  obtain ⟨hm,hn,hbk⟩ := Host.configuration_positive c hc
  obtain ⟨hmSmall,hnSmall,hkSmall⟩ := configuration_small c hc
  have tiles : 0<Host.cdiv h.shape.chunk c.N := by
    have := Geometry.cdiv_covers (n:=h.shape.chunk) hn
    by_contra hz
    have : Host.cdiv h.shape.chunk c.N=0 := by omega
    simp_all
  have pm (p : Geometry.PID h c) : Geometry.p0 p/Host.cdiv h.shape.chunk c.N<Host.cdiv h.shape.chunk c.M := by
    apply (Nat.div_lt_iff_lt_mul tiles).2
    exact (p ⟨0, by change 0<3; decide⟩).isLt
  have pn (p : Geometry.PID h c) : Geometry.p0 p%Host.cdiv h.shape.chunk c.N<Host.cdiv h.shape.chunk c.N :=
    Nat.mod_lt _ tiles
  constructor
  · intro p
    apply block_base h.shape.seqlen h.shape.chunk _ hchunk
    exact (Nat.div_lt_iff_lt_mul hg).2 (Geometry.p2_lt p)
  · intro p
    rw [←BitVec.ofNat_mul]
    have one : (1 : BitVec 64)=BitVec.ofNat 64 1 := rfl
    rw [one, ←BitVec.ofNat_add, ←BitVec.ofNat_mul]
    apply compare64
    · have := block_base h.shape.chunk c.N _ hn (pn p); omega
    · have := block_base h.shape.chunk c.M _ hm (pm p)
      have heq : (Geometry.p0 p / Host.cdiv h.shape.chunk c.N+1)*c.M =
        Geometry.p0 p / Host.cdiv h.shape.chunk c.N*c.M+c.M := by ring
      omega
  · intro p i limit hlimit
    exact mask_exact h.shape.chunk c.M _ i limit (by omega) hm hmSmall (pm p) i.isLt hlimit
  · intro p i limit hlimit
    exact mask_exact h.shape.chunk c.N _ i limit (by omega) hn hnSmall (pn p) i.isLt hlimit
  · intro k hk' i
    exact remainder_mask h.shape.K c.K k i (by omega) hbk hkSmall hk' i.isLt

theorem chunk_limit_native (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : AllocatedSizes h.shape) (p : Geometry.PID h c) :
    BitVec.ofNat 64 h.shape.seqlen -
      BitVec.ofNat 64 (Geometry.p2 p/h.shape.groups)*BitVec.ofNat 64 h.shape.chunk =
        BitVec.ofNat 64 (h.shape.seqlen-Geometry.p2 p/h.shape.groups*h.shape.chunk) ∧
    (BitVec.ofNat 64 (chunkLimit h c p)).toInt=chunkLimit h c p := by
  have base := (source_indices h c hc sizes).chunkBase p
  obtain ⟨hseq,hchunk,_⟩ := dimensions_bounded h.shape sizes
  constructor
  · rw [←BitVec.ofNat_mul]
    exact BitVec.ofNat_sub_ofNat_of_le _ _ (by omega) (by omega)
  · apply signed64
    have := Nat.min_le_left h.shape.chunk (h.shape.seqlen-Geometry.p2 p/h.shape.groups*h.shape.chunk)
    unfold chunkLimit
    omega

#axiomsClean dimensions_bounded
#axiomsClean source_indices
#axiomsClean chunk_limit_native
#axiomsClean active_address
#axiomsClean active_offset
end VeriTile.Bench.Repairs.BmmChunkFwd.Finite

#guard_msgs (drop info) in
#auditModuleAxioms
