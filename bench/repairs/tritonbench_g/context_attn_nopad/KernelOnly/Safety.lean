import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Allocation

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.Safety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open Checked
noncomputable section
set_option maxHeartbeats 3000000

abbrev Word := BitVec 64
def word (n : Nat) : Word := BitVec.ofNat 64 n

theorem word_exact {n : Nat} (hn : n < 2^63) : (word n).toInt = (n : Int) := by
  rw [word, ← BitVec.ofInt_natCast]
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · norm_num
  · exact_mod_cast hn

/-- Signed int32 load followed by the source's explicit int64 extension. -/
def metadataWord (x : Int) : Word := (BitVec.ofInt 32 x).signExtend 64

theorem metadataWord_exact {n : Nat} (hn : n < 2^31) :
    metadataWord (n : Int) = word n := by
  unfold metadataWord BitVec.signExtend word
  rw [BitVec.toInt_ofInt_eq_self (by decide) (by norm_num) (by exact_mod_cast hn)]
  exact BitVec.ofInt_natCast _ _


def timeWord (block lane : Nat) : Word := word block * word 128 + word lane

def queryWord (t : Tensor3) (start block lane head dim : Nat) : Word :=
  (word start + timeWord block lane) * word t.rowStride +
    word head * word t.headStride + word dim * word t.dimStride

/-- Exact source K/V address association: construct off_k/off_v, then advance
k_ptrs/v_ptrs by (start+start_n)*rowStride. All arithmetic is finite-width. -/
def keyWord (t : Tensor3) (start start_n lane head dim : Nat) : Word :=
  (word lane * word t.rowStride + word head * word t.headStride +
    word dim * word t.dimStride) + (word start + word start_n) * word t.rowStride

theorem timeWord_eq (block lane : Nat) : timeWord block lane = word (block*128+lane) := by
  simp only [timeWord, word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add]

theorem queryWord_eq (t : Tensor3) (start block lane head dim : Nat) :
    queryWord t start block lane head dim = word (t.offset (start+(block*128+lane)) head dim) := by
  simp only [queryWord, timeWord_eq, word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, Tensor3.offset]

theorem keyWord_eq (t : Tensor3) (start start_n lane head dim : Nat) :
    keyWord t start start_n lane head dim = word (t.offset (start+start_n+lane) head dim) := by
  simp only [keyWord, word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, Tensor3.offset]
  congr 1
  ring

def loopWord (block length : Nat) : Word :=
  word (if block*128 < length then 1 else 0) * (word block + word 1) * word 128

theorem loopWord_eq (block length : Nat) :
    loopWord block length = word (if block*128 < length then (block+1)*128 else 0) := by
  unfold loopWord
  split <;> simp [word, ← BitVec.ofNat_add, ← BitVec.ofNat_mul]

theorem padded_row (c : Call) (r : Regions) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c r extent) (limit : c.maxInputLen≤requestedMax)
    (b block lane : Nat) (hb : b < c.batches) (hblock : block < c.blocks) (hlane : lane < 128) :
    c.start b+(block*128+lane) < c.q.rows+requestedMax+128 := by
  have capacity := (g.2.2.2.2.2.2.1 ⟨b,hb⟩).1
  change c.start b+c.length b≤c.q.rows at capacity
  change block < (c.maxInputLen+127)/128 at hblock
  omega

/-- All formed offsets, including false-mask lanes, are bounded by the
actual source guard. No allocator or host-success fact is a premise. -/
theorem formed_offset (c : Call) (r : Regions) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c r extent) (limit : c.maxInputLen≤requestedMax)
    (v : View) (h : formedBudget c requestedMax v)
    (b block lane head dim : Nat) (hb : b < c.batches) (ht : block < c.blocks)
    (hl : lane < 128) (hh : head < c.q.heads) (hd : dim < c.q.dims) :
    c.q.offset (c.start b+(block*128+lane)) head dim < 2^61 := by
  have row := padded_row c r extent requestedMax g limit b block lane hb ht hl
  have rowBound := Nat.mul_le_mul_right (c.q.heads*c.q.dims)
    (show c.start b+(block*128+lane)≤c.q.rows+requestedMax+128-1 by omega)
  have headBound := Nat.mul_le_mul_right c.q.dims (show head≤c.q.heads-1 by omega)
  have dimBound : dim≤c.q.dims-1 := by omega
  have canonical := g.2.2.2.2.2.1
  simp only [Tensor3.offset,canonical.1,canonical.2.1,canonical.2.2,Nat.mul_one]
  have total := h.1
  omega

theorem all_formed_exact (c : Call) (r : Regions) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c r extent) (limit : c.maxInputLen≤requestedMax)
    (v : View) (h : formedBudget c requestedMax v)
    (b block iter lane head dim : Nat) (hb : b < c.batches) (ht : block < c.blocks)
    (hi : iter≤block) (hl : lane < 128) (hh : head < c.q.heads) (hd : dim < c.q.dims) :
    (queryWord c.q (c.start b) block lane head dim).toInt=
      (c.q.offset (c.start b+(block*128+lane)) head dim : Int) ∧
    (keyWord c.q (c.start b) (iter*128) lane head dim).toInt=
      (c.q.offset (c.start b+iter*128+lane) head dim : Int) ∧
    (word (4*c.q.offset (c.start b+(block*128+lane)) head dim)).toInt=
      (4*c.q.offset (c.start b+(block*128+lane)) head dim : Nat) ∧
    (word (4*c.q.offset (c.start b+iter*128+lane) head dim)).toInt=
      (4*c.q.offset (c.start b+iter*128+lane) head dim : Nat) := by
  have q := formed_offset c r extent requestedMax g limit v h b block lane head dim hb ht hl hh hd
  have k := formed_offset c r extent requestedMax g limit v h b iter lane head dim hb (lt_of_le_of_lt hi ht) hl hh hd
  rw [← Nat.add_assoc] at k
  refine ⟨?_,?_,word_exact (by omega),word_exact (by omega)⟩
  · rw [queryWord_eq]; exact word_exact (by omega)
  · rw [keyWord_eq]; exact word_exact (by omega)

/-- Finite native address/control expressions project to the mathematical
indices used by both interpretations of the one shared source. -/
structure Projects (r : Request) (heap : ByteMemory) (idx : GridIndex r.call.grid) : Prop where
  batch : (word (r.call.launch idx).batch).toInt=((r.call.launch idx).batch : Int)
  head : (word (r.call.launch idx).head).toInt=((r.call.launch idx).head : Int)
  block : (word (r.call.launch idx).block).toInt=((r.call.launch idx).block : Int)
  start : metadataWord (startValues r heap (r.call.launch idx).batch)=word (r.call.start (r.call.launch idx).batch)
  length : metadataWord (lengthValues r heap (r.call.launch idx).batch)=word (r.call.length (r.call.launch idx).batch)
  time : ∀ lane : Fin 128,(timeWord (r.call.launch idx).block lane.val).toInt=
    (((r.call.launch idx).block*128+lane.val : Nat) : Int)
  keyTime : ∀ iter,iter≤(r.call.launch idx).block → ∀ lane : Fin 128,
    (word (iter*128)+word lane.val).toInt=((iter*128+lane.val : Nat) : Int)
  channel : ∀ d : Fin r.call.q.dims,(word d.val).toInt=(d.val : Int)
  loop : (loopWord (r.call.launch idx).block (r.call.length (r.call.launch idx).batch)).toInt=
    ((if (r.call.launch idx).block*128 < r.call.length (r.call.launch idx).batch
      then ((r.call.launch idx).block+1)*128 else 0 : Nat) : Int)
  formed : ∀ t∈[r.call.q,r.call.k,r.call.v,r.call.o],∀ iter,iter≤(r.call.launch idx).block →
    ∀ (lane : Fin 128) (d : Fin r.call.q.dims),
    (queryWord t (r.call.start (r.call.launch idx).batch) (r.call.launch idx).block lane.val
      (r.call.launch idx).head d.val).toInt=
      (t.offset (r.call.start (r.call.launch idx).batch+((r.call.launch idx).block*128+lane.val)) (r.call.launch idx).head d.val : Int) ∧
    (keyWord t (r.call.start (r.call.launch idx).batch) (iter*128) lane.val (r.call.launch idx).head d.val).toInt=
      (t.offset (r.call.start (r.call.launch idx).batch+iter*128+lane.val) (r.call.launch idx).head d.val : Int) ∧
    (word (4*t.offset (r.call.start (r.call.launch idx).batch+((r.call.launch idx).block*128+lane.val)) (r.call.launch idx).head d.val)).toInt=
      ((4*t.offset (r.call.start (r.call.launch idx).batch+((r.call.launch idx).block*128+lane.val)) (r.call.launch idx).head d.val : Nat) : Int) ∧
    (word (4*t.offset (r.call.start (r.call.launch idx).batch+iter*128+lane.val) (r.call.launch idx).head d.val)).toInt=
      ((4*t.offset (r.call.start (r.call.launch idx).batch+iter*128+lane.val) (r.call.launch idx).head d.val : Nat) : Int)

theorem projects (r : Request) (heap : ByteMemory) (h : Contract r heap) (idx : GridIndex r.call.grid) :
    Projects r heap idx := by
  have hb := launch_batch_lt r.call idx
  have hh := launch_head_lt r.call idx
  have ht := launch_block_lt r.call idx
  have native := h.checks.2.2.1
  have batchBound := native.2.2.1
  have headBound := native.2.2.2.1
  have blockBound := native.2.2.2.2.1
  have shape := (layout_memory r heap h).shape
  refine ⟨word_exact (by omega),word_exact (by omega),word_exact (by omega),?_,?_,?_,?_,?_,?_,?_⟩
  · rw [(h.checks.2.2.2.1 ⟨_,hb⟩).1]
    exact metadataWord_exact (shape.int32Start _ hb)
  · rw [(h.checks.2.2.2.1 ⟨_,hb⟩).2]
    exact metadataWord_exact (shape.int32Length _ hb)
  · intro lane
    rw [timeWord_eq]
    apply word_exact
    have := lane.isLt
    omega
  · intro iter hi lane
    rw [word,word,←BitVec.ofNat_add]
    apply word_exact
    have := lane.isLt
    omega
  · intro d
    apply word_exact
    have := d.isLt
    rcases shape.width with hw|hw|hw|hw <;> omega
  · rw [loopWord_eq]
    apply word_exact
    split <;> omega
  · intro t member iter hi lane d
    have keq := h.checks.2.1.2.2.1
    have veq := h.checks.2.1.2.2.2.1
    have oeq := h.checks.2.1.2.2.2.2.1
    have eq : t=r.call.q := by simpa [keq,veq,oeq] using member
    subst t
    exact all_formed_exact r.call regions (extent r) r.requestedMax h.checks.2.1 native.2.1
      (views r).q (native.2.2.2.2.2.1 _ (by simp)) _ _ _ _ _ _ hb ht hi lane.isLt hh d.isLt

#axiomsClean projects
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.Safety
