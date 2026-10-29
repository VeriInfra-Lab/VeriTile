import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideDomain

/-! Int32/int64 metadata ABI and complete finite-word projection. The address
polynomial identities are reused unchanged from NativeProgram. The wider loop
bound is derived from output allocation/injectivity in WideDomain, not assumed
as a new public input restriction. Metadata dtypes may differ between roles. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.WideNative
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Formula Program Host HostPlan NativeProgram
noncomputable section

inductive MetadataType where
  | int32 | int64
  deriving DecidableEq

def MetadataType.bits : MetadataType → Nat
  | .int32 => 32 | .int64 => 64

def MetadataType.bytes : MetadataType → Nat
  | .int32 => 4 | .int64 => 8

structure MetadataTypes where
  start : MetadataType
  length : MetadataType

/-- Nonnegative logical metadata must fit its selected signed storage dtype.
This is the ordinary representation contract, not a selected safe index bound. -/
structure MetadataFits (c : Call) (types : MetadataTypes) : Prop where
  start : ∀ b, b < c.batches → c.start b < 2^(types.start.bits-1)
  length : ∀ b, b < c.batches → c.length b < 2^(types.length.bits-1)

/-- The two integer allocations use their own element sizes. In particular
int64 metadata is not assigned the FP32 tensors' four-byte pointer scale. -/
structure MetadataAlloc (r : Regions) (A : FlatAlloc) (types : MetadataTypes) : Prop where
  startBytes : types.start.bytes * A.extent r.start ≤ 2^63
  lengthBytes : types.length.bytes * A.extent r.sequence ≤ 2^63

theorem metadata_byte_offset (dtype : MetadataType) {extent offset : Nat}
    (hbytes : dtype.bytes * extent ≤ 2^63) (hoff : offset < extent) :
    (word offset * word dtype.bytes).toInt = ((offset*dtype.bytes : Nat) : Int) := by
  rw [word,word,← BitVec.ofNat_mul]
  apply word_exact
  have hpos : 0 < dtype.bytes := by cases dtype <;> decide
  have hlt := Nat.mul_lt_mul_of_pos_right hoff hpos
  have hb : extent*dtype.bytes ≤ 2^63 := by simpa [Nat.mul_comm] using hbytes
  exact lt_of_lt_of_le hlt hb

theorem metadata_byte_offsets {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (types : MetadataTypes) (alloc : MetadataAlloc r A types)
    (b : Nat) (hb : b < c.batches) :
    (word b * word types.start.bytes).toInt = ((b*types.start.bytes : Nat) : Int) ∧
    (word b * word types.length.bytes).toInt = ((b*types.length.bytes : Nat) : Int) :=
  ⟨metadata_byte_offset types.start alloc.startBytes (lt_of_lt_of_le hb hm.startExtent),
    metadata_byte_offset types.length alloc.lengthBytes (lt_of_lt_of_le hb hm.lengthExtent)⟩

def loadWord (dtype : MetadataType) (x : Int) : Word :=
  (BitVec.ofInt dtype.bits x).signExtend 64

theorem loadWord_exact (dtype : MetadataType) {n : Nat} (hn : n < 2^(dtype.bits-1)) :
    loadWord dtype (n : Int) = word n := by
  cases dtype with
  | int32 => exact metadataWord_exact hn
  | int64 =>
    unfold loadWord MetadataType.bits BitVec.signExtend word
    rw [BitVec.toInt_ofInt_eq_self (by decide) (by norm_num) (by exact_mod_cast hn)]
    exact BitVec.ofInt_natCast _ _

structure Projects (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (types : MetadataTypes) (j : Job c) : Prop where
  byteOffset : ∀ rg addr, addr < A.extent rg →
    (word addr * word 4).toInt = ((addr*4 : Nat) : Int)
  batch : (pidWord j.1).toInt = ((global j.1).val : Int)
  head : (pidWord j.2.1).toInt = ((global j.2.1).val : Int)
  block : (pidWord j.2.2).toInt = ((global j.2.2).val : Int)
  start : loadWord types.start ((inputs s r).start (global j.1).val) = word (c.start (global j.1).val)
  length : loadWord types.length ((inputs s r).sequence (global j.1).val) = word (c.length (global j.1).val)
  time : ∀ lane : Fin 128, (timeWord (global j.2.2).val lane.val).toInt =
    ((global j.2.2).val*128+lane.val : Nat)
  keyTime : ∀ iter, iter ≤ (global j.2.2).val → ∀ lane : Fin 128,
    (word (iter*128) + word lane.val).toInt = ((iter*128+lane.val : Nat) : Int)
  channel : ∀ d : Fin c.q.dims, (word d.val).toInt = (d.val : Int)
  loop : (loopWord (global j.2.2).val (c.length (global j.1).val)).toInt =
    ((if (global j.2.2).val*128 < c.length (global j.1).val then
      ((global j.2.2).val+1)*128 else 0 : Nat) : Int)
  query : ∀ (lane : Fin 128) (d : Fin c.q.dims),
    (global j.2.2).val*128+lane.val < c.length (global j.1).val →
    (queryWord c.q (c.start (global j.1).val) (global j.2.2).val lane.val
      (global j.2.1).val d.val).toInt =
      (c.q.offset (c.start (global j.1).val+((global j.2.2).val*128+lane.val))
        (global j.2.1).val d.val : Int)
  key : ∀ start_n (lane : Fin 128) (d : Fin c.q.dims),
    start_n+lane.val < c.length (global j.1).val →
    (keyWord c.k (c.start (global j.1).val) start_n lane.val (global j.2.1).val d.val).toInt =
      (c.k.offset (c.start (global j.1).val+start_n+lane.val) (global j.2.1).val d.val : Int)
  value : ∀ start_n (lane : Fin 128) (d : Fin c.q.dims),
    start_n+lane.val < c.length (global j.1).val →
    (keyWord c.v (c.start (global j.1).val) start_n lane.val (global j.2.1).val d.val).toInt =
      (c.v.offset (c.start (global j.1).val+start_n+lane.val) (global j.2.1).val d.val : Int)
  output : ∀ (lane : Fin 128) (d : Fin c.q.dims),
    (global j.2.2).val*128+lane.val < c.length (global j.1).val →
    (queryWord c.o (c.start (global j.1).val) (global j.2.2).val lane.val
      (global j.2.1).val d.val).toInt =
      (c.o.offset (c.start (global j.1).val+((global j.2.2).val*128+lane.val))
        (global j.2.1).val d.val : Int)

theorem projects {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (ha : ABI c A) (types : MetadataTypes)
    (fits : MetadataFits c types) (hb : c.blocks ≤ 2^54) (j : Job c) :
    Projects c r A s types j := by
  have hbatch := (global j.1).isLt
  have hhead := (global j.2.1).isLt
  have hblock := (global j.2.2).isLt
  norm_num at hb
  have hb64 : (global j.1).val < 2^63 :=
    lt_of_lt_of_le hbatch (le_trans hm.startExtent (ha.extent r.start))
  have hh64 : (global j.2.1).val < 2^63 := lt_of_lt_of_le hhead ha.heads
  have ht64 : (global j.2.2).val < 2^63 := by omega
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro rg addr had
    rw [word, word, ← BitVec.ofNat_mul]
    apply word_exact
    have := ha.bytes rg
    omega
  · rw [pidWord_eq]; exact word_exact hb64
  · rw [pidWord_eq]; exact word_exact hh64
  · rw [pidWord_eq]; exact word_exact ht64
  · rw [hm.start _ hbatch]; exact loadWord_exact types.start (fits.start _ hbatch)
  · rw [hm.length _ hbatch]; exact loadWord_exact types.length (fits.length _ hbatch)
  · intro lane
    rw [timeWord_eq]
    apply word_exact
    have := lane.isLt
    omega
  · intro iter hi lane
    rw [word, word, ← BitVec.ofNat_add]
    apply word_exact
    have := lane.isLt
    omega
  · intro d
    apply word_exact
    have := d.isLt
    rcases hm.shape.width with hw | hw | hw | hw <;> omega
  · rw [loopWord_eq]
    apply word_exact
    split <;> omega
  · intro lane d hl
    rw [queryWord_eq]
    apply word_exact
    apply lt_of_lt_of_le _ (ha.extent r.query)
    apply hm.qBounded
    · have := hm.shape.qRows _ hbatch; omega
    · exact hhead
    · exact d.isLt
  · intro start_n lane d hl
    rw [keyWord_eq]
    apply word_exact
    apply lt_of_lt_of_le _ (ha.extent r.key)
    apply hm.kBounded
    · have := hm.shape.kRows _ hbatch; omega
    · exact lt_of_lt_of_le hhead hm.shape.kHeads
    · simpa only [hm.shape.kDims] using d.isLt
  · intro start_n lane d hl
    rw [keyWord_eq]
    apply word_exact
    apply lt_of_lt_of_le _ (ha.extent r.value)
    apply hm.vBounded
    · have := hm.shape.vRows _ hbatch; omega
    · exact lt_of_lt_of_le hhead hm.shape.vHeads
    · simpa only [hm.shape.vDims] using d.isLt
  · intro lane d hl
    rw [queryWord_eq]
    apply word_exact
    apply lt_of_lt_of_le _ (ha.extent r.out)
    apply hm.oBounded
    · have := hm.shape.oRows _ hbatch; omega
    · exact lt_of_lt_of_le hhead hm.shape.oHeads
    · exact lt_of_lt_of_le d.isLt hm.shape.oDims


/-- The final key-loop increment is evaluated too, even though its next test
exits. Its exactness follows from the same allocation-derived rounded bound. -/
theorem loop_backedge {c : Call} (hb : c.blocks ≤ 2^54) (j : Job c)
    (iter : Nat) (hi : iter ≤ (global j.2.2).val) :
    (word (iter*128) + word 128).toInt = ((iter*128+128 : Nat) : Int) := by
  rw [word,word,← BitVec.ofNat_add]
  exact word_exact (WideDomain.key_lane_and_backedge hb (global j.2.2).isLt hi (by decide : 0 < 128)).2

def capMemory {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (maximum : Nat) :
    WideDomain.Memory (NativeIndex.capCall c maximum) r A s :=
  ⟨⟨hm.shape.width,hm.shape.kDims,hm.shape.vDims,hm.shape.oDims,
      hm.shape.kHeads,hm.shape.vHeads,hm.shape.oHeads,hm.shape.qRows,
      hm.shape.kRows,hm.shape.vRows,hm.shape.oRows,hm.shape.intervals,hm.shape.outputInjective⟩,
    hm.start,hm.length,hm.startExtent,hm.lengthExtent,hm.qBounded,hm.kBounded,
    hm.vBounded,hm.oBounded,hm.outQuery,hm.outKey,hm.outValue,hm.outStart,hm.outSequence⟩

def capFits {c : Call} {types : MetadataTypes} (fits : MetadataFits c types) (maximum : Nat) :
    MetadataFits (NativeIndex.capCall c maximum) types := ⟨fits.start,fits.length⟩

/-- Every actually launched job has a head. With no heads this statement is
vacuous, so arbitrary representable inactive metadata is not narrowed. -/
theorem complete_projection {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (ha : ABI c A) (types : MetadataTypes)
    (fits : MetadataFits c types) (j : Job (HostPlan.effective c r s)) :
    Projects (HostPlan.effective c r s) r A s types j ∧
    (∀ iter, iter ≤ (global j.2.2).val →
      (word (iter*128) + word 128).toInt = ((iter*128+128 : Nat) : Int)) := by
  have hh : 0 < c.q.heads := by
    have h := (global j.2.1).isLt
    change (global j.2.1).val < c.q.heads at h
    omega
  have hb := WideDomain.effective_blocks_bound hm ha hh
  exact ⟨projects (capMemory hm _) ⟨ha.heads,ha.bytes⟩ types (capFits fits _) hb j,
    fun iter hi => loop_backedge hb j iter hi⟩

theorem nativePids_eq {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    {types : MetadataTypes} {j : Job c} (hp : Projects c r A s types j) :
    NativeProgram.nativePids c j = (index c j).toPids := by
  funext ax
  rcases ax with _ | (_ | (_ | ax))
  · simp [NativeProgram.nativePids,hp.batch,index,GridIndex.toPids,Call.grid,Grid.rank,Call.index]
  · simp [NativeProgram.nativePids,hp.head,index,GridIndex.toPids,Call.grid,Grid.rank,Call.index]
  · simp [NativeProgram.nativePids,hp.block,index,GridIndex.toPids,Call.grid,Grid.rank,Call.index]
    rfl
  · simp [NativeProgram.nativePids,index,GridIndex.toPids,Call.grid,Grid.rank,Call.index]

#print axioms complete_projection
end
end VeriTile.Bench.Repairs.ContextAttnNopad.WideNative
