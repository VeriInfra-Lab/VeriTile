import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeInterface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 2000000

/-- A masked native memory event has no address requirement on false lanes.
The unbounded projection uses the same byte memory and an exact active offset. -/
def maskedLoad {α : Type} (memory : Nat → α) (zero : α) (mask : Bool) (pointer : BitVec 64) : α :=
  if mask then memory pointer.toNat else zero

def maskedStore {α : Type} (memory : Nat → α) (value : α) (mask : Bool) (pointer : BitVec 64) : Nat → α :=
  if mask then Function.update memory pointer.toNat value else memory

theorem masked_load_exact {α : Type} (memory : Nat → α) (zero : α)
    (mask : Bool) (pointer : BitVec 64) (active : Prop) [Decidable active] (address : Nat)
    (hm : mask = Bool.true ↔ active) (hp : active → pointer.toNat = address) :
    maskedLoad memory zero mask pointer = if active then memory address else zero := by
  by_cases ha : active
  · simp [maskedLoad,hm.mpr ha,ha,hp ha]
  · have hfalse : mask = Bool.false := by cases mask <;> simp_all
    simp [maskedLoad,hfalse,ha]

theorem masked_store_exact {α : Type} (memory : Nat → α) (value : α)
    (mask : Bool) (pointer : BitVec 64) (active : Prop) [Decidable active] (address : Nat)
    (hm : mask = Bool.true ↔ active) (hp : active → pointer.toNat = address) :
    maskedStore memory value mask pointer = if active then Function.update memory address value else memory := by
  by_cases ha : active
  · simp [maskedStore,hm.mpr ha,ha,hp ha]
  · have hfalse : mask = Bool.false := by cases mask <;> simp_all
    simp [maskedStore,hfalse,ha]

theorem disjoint_intervals (a n b m x y : Nat) (apart : a+n ≤ b ∨ b+m ≤ a)
    (hx : a ≤ x ∧ x < a+n) (hy : b ≤ y ∧ y < b+m) : x ≠ y := by
  rcases apart with h | h <;> omega

/-- Physical fresh scratch bytes are disjoint from every read allocation.
Read inputs may alias one another, and caller output is outside this object. -/
structure SeparateBytes (storage : Storage) (buffers : Buffers) : Prop where
  data : ∀ r, r ≠ .output → ∀ x y,
    (buffers.data .output).base ≤ x → x < (buffers.data .output).base+dataBytes storage*(buffers.data .output).extent →
    (buffers.data r).base ≤ y → y < (buffers.data r).base+dataBytes storage*(buffers.data r).extent → x ≠ y
  metadata : ∀ r x y,
    (buffers.data .output).base ≤ x → x < (buffers.data .output).base+dataBytes storage*(buffers.data .output).extent →
    (buffers.metadataBuffer r).base ≤ y →
      y < (buffers.metadataBuffer r).base+(buffers.metadataBuffer r).width.bytes*(buffers.metadataBuffer r).extent → x ≠ y

theorem separate_bytes (c : Host.Call) (storage : Storage) (buffers : Buffers)
    (ready : Ready c storage buffers) : SeparateBytes storage buffers := by
  constructor
  · intro r hr x y hxlo hxhi hylo hyhi
    exact disjoint_intervals _ _ _ _ x y (ready.fresh_data r hr) ⟨hxlo,hxhi⟩ ⟨hylo,hyhi⟩
  · intro r x y hxlo hxhi hylo hyhi
    exact disjoint_intervals _ _ _ _ x y (ready.fresh_metadata r) ⟨hxlo,hxhi⟩ ⟨hylo,hyhi⟩

/-- Point coordinates are exactly the PIDs used by the normalized program. -/
theorem point_source (c : Host.Call) (idx : GridIndex c.grid) :
    (pointOfGrid c idx).batch.val = (c.launchAt idx).batch ∧
    (pointOfGrid c idx).head.val = (c.launchAt idx).head ∧
    (pointOfGrid c idx).block.val = (c.launchAt idx).block := ⟨rfl,rfl,rfl⟩

/-- The independent formula visits the same complete key tiles as the signed
native loop, including inactive blocks and the final partial key tile. -/
theorem normalized_count (c : Host.Call) (p : Point c) (a : Formula.Inputs)
    (length : a.sequence p.batch = (c.length p.batch : Int)) :
    Formula.count c.layout ⟨p.block,p.batch,p.head⟩ a = chunks p.block (c.length p.batch) := by
  simp only [Formula.count,Formula.sequence,c.blockM,c.blockN,length]
  have hc : ((p.block.val : Int)*128 < (c.length p.batch : Nat)) ↔ p.block.val*128 < c.length p.batch := by omega
  simp [hc,chunks]

/-- For the actual source grid, one correspondence contains every ABI route,
loop test/backedge, padded predicate, staged pointer, active complete word,
metadata width projection and masked read/write. No numerical law appears. -/
structure Correspondence (c : Host.Call) (storage : Storage) (buffers : Buffers) : Prop where
  pids : ∀ p : Point c,
    (fromI32 p.batch).toInt = (p.batch.val : Int) ∧
    (fromI32 p.head).toInt = (p.head.val : Int) ∧
    (fromI32 p.block).toInt = (p.block.val : Int)
  strides : ∀ r : DataRole, (scalarABI (r.rowStride c)).toInt = (r.rowStride c : Int) ∧
    (scalarABI (r.headStride c)).toInt = (r.headStride c : Int) ∧
    (scalarABI (r.dimStride c)).toInt = (r.dimStride c : Int)
  gqa : ∀ p : Point c, (fromI32 p.head).sdiv (scalarABI c.layout.kvGroup) = word (p.head/c.layout.kvGroup)
  metadata : ∀ r (p : Point c),
    WordAt (buffers.metadataBuffer r).toBuffer (buffers.metadataBuffer r).width.bytes p.batch (metaPointer buffers r p.batch) ∧
    ((buffers.metadataBuffer r).width.widen ((buffers.metadataBuffer r).width.raw (r.value c p.batch))).toInt = (r.value c p.batch : Int)
  sourceCount : ∀ (p : Point c) (a : Formula.Inputs),
    a.sequence p.batch = (c.length p.batch : Int) →
      Formula.count c.layout ⟨p.block,p.batch,p.head⟩ a = chunks p.block (c.length p.batch)
  rowValues : ∀ (p : Point c) lane, lane < 128 →
    (rowBits p.block lane).toInt = (p.block.val*128+lane : Nat)
  keyValues : ∀ (p : Point c) step lane, step < chunks p.block (c.length p.batch) → lane < 128 →
    (keyBits step lane).toInt = (step*128+lane : Nat)
  dimensions : ∀ d, d < c.layout.D → (fromI32 d).toInt = (d : Int)
  loop : ∀ (p : Point c) step, step ≤ chunks p.block (c.length p.batch) →
    (indexBits step).toInt = (step*128 : Nat) ∧
    (stopBits (buffers.metadataBuffer .length).width p.block (c.length p.batch)).toInt =
      (chunks p.block (c.length p.batch)*128 : Nat) ∧
    ((indexBits step).slt (stopBits (buffers.metadataBuffer .length).width p.block (c.length p.batch)) = Bool.true ↔
      step < chunks p.block (c.length p.batch)) ∧ (indexBits step).toInt % 128 = 0
  queryModular : ∀ r, r = .query ∨ r = .output → ∀ (p : Point c) lane d,
    lane < 128 → d < c.layout.D →
    queryPointer c storage buffers r p lane d = word ((buffers.data r).base+
      dataBytes storage*elementOffset c r p.batch p.head (p.block*128+lane) d)
  keyModular : ∀ r, r = .key ∨ r = .value → ∀ (p : Point c) step lane d,
    lane < 128 → d < c.layout.D →
    keyPointer c storage buffers r p step lane d = word ((buffers.data r).base+
      dataBytes storage*elementOffset c r p.batch p.head (step*128+lane) d)
  queryMask : ∀ (p : Point c) lane, lane < 128 →
    ((rowBits p.block lane).slt (metadataABI (buffers.metadataBuffer .length).width (c.length p.batch)) = Bool.true ↔
      p.block*128+lane < c.length p.batch)
  keyMask : ∀ (p : Point c) step lane, step < chunks p.block (c.length p.batch) → lane < 128 →
    ((keyBits step lane).slt (metadataABI (buffers.metadataBuffer .length).width (c.length p.batch)) = Bool.true ↔
      step*128+lane < c.length p.batch)
  queryWord : ∀ r, r = .query ∨ r = .output → ∀ (p : Point c) lane d,
    lane < 128 → d < c.layout.D → p.block*128+lane < c.length p.batch →
    WordAt (buffers.data r) (dataBytes storage) (elementOffset c r p.batch p.head (p.block*128+lane) d)
      (queryPointer c storage buffers r p lane d)
  keyWord : ∀ r, r = .key ∨ r = .value → ∀ (p : Point c) step lane d,
    lane < 128 → d < c.layout.D → step*128+lane < c.length p.batch →
    WordAt (buffers.data r) (dataBytes storage) (elementOffset c r p.batch p.head (step*128+lane) d)
      (keyPointer c storage buffers r p step lane d)
  window : ∀ (p : Point c) step i j, step < chunks p.block (c.length p.batch) → i < 128 → j < 128 →
    (((keyBits step j).sle (rowBits p.block i) = Bool.true ∧
      (rowBits p.block i-keyBits step j).slt (windowABI c.layout.window) = Bool.true) ↔
      ((step*128+j : Int) ≤ (p.block*128+i : Nat) ∧
        (step*128+j : Int) > (p.block*128+i : Nat)-c.layout.window))
  separate : SeparateBytes storage buffers
  signedPointer : ∀ (bytes : Nat) (offset : BitVec 64), sword ((bytes : Int)*offset.toInt) = word bytes*offset

theorem correspondence (c : Host.Call) (storage : Storage) (buffers : Buffers)
    (ready : Ready c storage buffers) : Correspondence c storage buffers := by
  refine ⟨?_,?_,?_,?_,normalized_count c,?_,?_,?_,?_,query_pointer_modular c storage buffers ready,
    key_pointer_modular c storage buffers ready,?_,?_,query_word c storage buffers ready,
    key_word c storage buffers ready,?_,separate_bytes c storage buffers ready,signed_byte_residue⟩
  · intro p
    have hp := ready.grid p
    simp only [fromI32_word p.batch hp.1,fromI32_word p.head hp.2.1,fromI32_word p.block hp.2.2]
    exact ⟨word_signed _ (by omega),word_signed _ (by omega),word_signed _ (by omega)⟩
  · intro r
    have h := ready.strides r
    simp only [scalarABI_word]
    exact ⟨word_signed _ h.1,word_signed _ h.2.1,word_signed _ h.2.2⟩
  · intro p
    exact signed_div _ _ (ready.grid p).2.1 ready.group
  · intro r p
    exact ⟨metadata_word_at c storage buffers ready r p,
      (metadata_load _ _ (ready.metadata r p.batch p.batch.isLt)).2⟩
  · intro p lane hl
    exact row_signed _ _ (ready.grid p).2.2 hl
  · intro p step lane ht hl
    exact key_signed _ _ _ _ (ready.grid p).2.2 ht hl
  · intro d hd
    have hdim := dim_bound c d hd
    rw [fromI32_word d (by omega)]
    exact word_signed d (by omega)
  · intro p step ht
    exact loop_exact _ _ _ _ (ready.grid p).2.2 (ready.metadata .length p.batch p.batch.isLt) ht
  · intro p lane hl
    exact row_mask _ _ _ _ (ready.grid p).2.2 (ready.metadata .length p.batch p.batch.isLt) hl
  · intro p step lane ht hl
    exact key_mask _ _ _ _ _ (ready.grid p).2.2 (ready.metadata .length p.batch p.batch.isLt) ht hl
  · intro p step i j ht hi hj
    exact window_mask _ _ _ _ _ _ (ready.grid p).2.2 ht hi hj

/-- Consequence at the actual source load site; arbitrary opaque values are
preserved, and a padded lane does not read any physical address. -/
theorem Correspondence.query_load {c : Host.Call} {storage : Storage} {buffers : Buffers}
    (h : Correspondence c storage buffers) {α : Type} (memory : Nat → α) (zero : α)
    (p : Point c) (lane d : Nat) (hl : lane < 128) (hd : d < c.layout.D) :
    maskedLoad memory zero ((rowBits p.block lane).slt (metadataABI (buffers.metadataBuffer .length).width (c.length p.batch)))
      (queryPointer c storage buffers .query p lane d) =
    if p.block*128+lane < c.length p.batch then memory ((buffers.data .query).base+
      dataBytes storage*elementOffset c .query p.batch p.head (p.block*128+lane) d) else zero := by
  apply masked_load_exact _ _ _ _ _ _ (h.queryMask p lane hl)
  intro ha
  exact (h.queryWord .query (Or.inl rfl) p lane d hl hd ha).1

theorem Correspondence.key_load {c : Host.Call} {storage : Storage} {buffers : Buffers}
    (h : Correspondence c storage buffers) {α : Type} (memory : Nat → α) (zero : α)
    (r : DataRole) (hr : r = .key ∨ r = .value) (p : Point c) (step lane d : Nat)
    (ht : step < chunks p.block (c.length p.batch)) (hl : lane < 128) (hd : d < c.layout.D) :
    maskedLoad memory zero ((keyBits step lane).slt (metadataABI (buffers.metadataBuffer .length).width (c.length p.batch)))
      (keyPointer c storage buffers r p step lane d) =
    if step*128+lane < c.length p.batch then memory ((buffers.data r).base+
      dataBytes storage*elementOffset c r p.batch p.head (step*128+lane) d) else zero := by
  apply masked_load_exact _ _ _ _ _ _ (h.keyMask p step lane ht hl)
  intro ha
  exact (h.keyWord r hr p step lane d hl hd ha).1

theorem Correspondence.output_store {c : Host.Call} {storage : Storage} {buffers : Buffers}
    (h : Correspondence c storage buffers) {α : Type} (memory : Nat → α) (value : α)
    (p : Point c) (lane d : Nat) (hl : lane < 128) (hd : d < c.layout.D) :
    maskedStore memory value ((rowBits p.block lane).slt (metadataABI (buffers.metadataBuffer .length).width (c.length p.batch)))
      (queryPointer c storage buffers .output p lane d) =
    if p.block*128+lane < c.length p.batch then Function.update memory ((buffers.data .output).base+
      dataBytes storage*elementOffset c .output p.batch p.head (p.block*128+lane) d) value else memory := by
  apply masked_store_exact _ _ _ _ _ _ (h.queryMask p lane hl)
  intro ha
  exact (h.queryWord .output (Or.inr rfl) p lane d hl hd ha).1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
