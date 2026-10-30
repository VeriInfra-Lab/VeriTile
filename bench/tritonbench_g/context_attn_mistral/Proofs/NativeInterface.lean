import bench.tritonbench_g.context_attn_mistral.Proofs.NativePointers

namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 2000000

inductive DataRole where
  | query | key | value | output
  deriving DecidableEq
inductive MetaRole where
  | start | length
  deriving DecidableEq

def DataRole.rowStride (r : DataRole) (c : Host.Call) : Nat := match r with
  | .query => c.layout.qRow | .key => c.layout.kRow
  | .value => c.layout.vRow | .output => c.layout.outRow
def DataRole.headStride (r : DataRole) (c : Host.Call) : Nat := match r with
  | .query => c.layout.qHead | .key => c.layout.kHead
  | .value => c.layout.vHead | .output => c.layout.outHead
def DataRole.dimStride (r : DataRole) (c : Host.Call) : Nat := match r with
  | .query => c.layout.qDim | .key => c.layout.kDim
  | .value => c.layout.vDim | .output => c.layout.outDim
def DataRole.head (r : DataRole) (c : Host.Call) (h : Nat) : Nat := match r with
  | .query | .output => h | .key | .value => h/c.layout.kvGroup

def dataBytes : Storage → Nat | .fp16 | .bf16 => 2 | .fp32 => 4
theorem dataBytes_pos (storage : Storage) : 0 < dataBytes storage := by cases storage <;> decide

def MetaRole.value (r : MetaRole) (c : Host.Call) (b : Nat) : Nat := match r with
  | .start => c.start b | .length => c.length b

structure Buffer where
  base : Nat
  extent : Nat
structure MetadataBuffer extends Buffer where
  width : MetadataWidth
structure Buffers where
  data : DataRole → Buffer
  metadataBuffer : MetaRole → MetadataBuffer

/-- An actual launch coordinate, equivalent to one member of c.grid. -/
structure Point (c : Host.Call) where
  batch : Fin c.batches
  head : Fin c.heads
  block : Fin c.blocks

def pointOfGrid (c : Host.Call) (idx : GridIndex c.grid) : Point c :=
  ⟨idx ⟨0,by simp [Host.Call.grid,Grid.rank]⟩,
   idx ⟨1,by simp [Host.Call.grid,Grid.rank]⟩,
   idx ⟨2,by simp [Host.Call.grid,Grid.rank]⟩⟩

def elementOffset (c : Host.Call) (r : DataRole) (b h row d : Nat) : Nat :=
  (c.start b+row)*r.rowStride c+r.head c h*r.headStride c+d*r.dimStride c

/-- Only ordinary launch ABI, shape, storage and fresh-allocation facts.
The grid clause applies to actual PIDs, so an empty launch is not restricted
by an irrelevant padded-row bound. No inactive pointer bound is required. -/
structure Ready (c : Host.Call) (storage : Storage) (buffers : Buffers) : Prop where
  grid : ∀ p : Point c, p.batch.val < 2^31 ∧ p.head.val < 2^31 ∧ p.block.val < 2^31
  strides : ∀ r : DataRole, r.rowStride c < 2^63 ∧ r.headStride c < 2^63 ∧ r.dimStride c < 2^63
  group : c.layout.kvGroup < 2^63
  group_positive : ∀ _p : Point c, 0 < c.layout.kvGroup
  group_shape : c.heads = c.kvHeads*c.layout.kvGroup
  metadata : ∀ r b, b < c.batches → (buffers.metadataBuffer r).width.Fits (r.value c b)
  metadata_extent : ∀ r, c.batches ≤ (buffers.metadataBuffer r).extent
  metadata_backing : ∀ r, (buffers.metadataBuffer r).base+(buffers.metadataBuffer r).width.bytes*(buffers.metadataBuffer r).extent ≤ 2^64
  backing : ∀ r, (buffers.data r).base+dataBytes storage*(buffers.data r).extent ≤ 2^64
  coordinates : ∀ r b h row d, b < c.batches → h < c.heads → row < c.length b → d < c.layout.D →
    elementOffset c r b h row d < (buffers.data r).extent
  fresh_data : ∀ r, r ≠ .output →
    (buffers.data .output).base+dataBytes storage*(buffers.data .output).extent ≤ (buffers.data r).base ∨
    (buffers.data r).base+dataBytes storage*(buffers.data r).extent ≤ (buffers.data .output).base
  fresh_metadata : ∀ r,
    (buffers.data .output).base+dataBytes storage*(buffers.data .output).extent ≤ (buffers.metadataBuffer r).base ∨
    (buffers.metadataBuffer r).base+(buffers.metadataBuffer r).width.bytes*(buffers.metadataBuffer r).extent ≤ (buffers.data .output).base

def queryPointer (c : Host.Call) (storage : Storage) (buffers : Buffers) (r : DataRole)
    (p : Point c) (lane d : Nat) : BitVec 64 :=
  addptr (word (buffers.data r).base) (dataBytes storage)
    (queryABI (buffers.metadataBuffer .start).width (c.start p.batch) p.block lane p.head d
      (r.rowStride c) (r.headStride c) (r.dimStride c))

def keyPointer (c : Host.Call) (storage : Storage) (buffers : Buffers) (r : DataRole)
    (p : Point c) (step lane d : Nat) : BitVec 64 :=
  keyABI (buffers.metadataBuffer .start).width (buffers.data r).base (dataBytes storage)
    (c.start p.batch) step lane p.head d c.layout.kvGroup (r.rowStride c) (r.headStride c) (r.dimStride c)

def metaPointer (buffers : Buffers) (r : MetaRole) (b : Nat) : BitVec 64 :=
  addptr (word (buffers.metadataBuffer r).base) (buffers.metadataBuffer r).width.bytes (fromI32 b)

/-- Explicit complete-word predicate; the interior-byte clauses also express
in-bounds access, including last words at the end of a backing allocation. -/
def WordAt (buffer : Buffer) (bytes offset : Nat) (pointer : BitVec 64) : Prop :=
  pointer.toNat = buffer.base+bytes*offset ∧
    ∀ j, j < bytes → (pointer+word j).toNat = buffer.base+bytes*offset+j ∧
      buffer.base ≤ buffer.base+bytes*offset+j ∧
      buffer.base+bytes*offset+j < buffer.base+bytes*buffer.extent

theorem dim_bound (c : Host.Call) (d : Nat) (hd : d < c.layout.D) : d < 128 := by
  have h := c.channels
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  omega

theorem query_pointer_modular (c : Host.Call) (storage : Storage) (buffers : Buffers)
    (ready : Ready c storage buffers) (r : DataRole) (hr : r = .query ∨ r = .output)
    (p : Point c) (lane d : Nat) (hl : lane < 128) (hd : d < c.layout.D) :
    queryPointer c storage buffers r p lane d =
      word ((buffers.data r).base+dataBytes storage*elementOffset c r p.batch p.head (p.block*128+lane) d) := by
  have hs := ready.metadata .start p.batch p.batch.isLt
  dsimp only [MetaRole.value] at hs
  have hp := ready.grid p
  rw [queryPointer,queryABI_modular _ _ _ _ _ _ _ _ _ hs hp.2.2 hl hp.2.1 (dim_bound c d hd),addptr_modular]
  rcases hr with rfl | rfl <;> rfl

theorem key_pointer_modular (c : Host.Call) (storage : Storage) (buffers : Buffers)
    (ready : Ready c storage buffers) (r : DataRole) (hr : r = .key ∨ r = .value)
    (p : Point c) (step lane d : Nat) (hl : lane < 128) (hd : d < c.layout.D) :
    keyPointer c storage buffers r p step lane d =
      word ((buffers.data r).base+dataBytes storage*elementOffset c r p.batch p.head (step*128+lane) d) := by
  have hs := ready.metadata .start p.batch p.batch.isLt
  dsimp only [MetaRole.value] at hs
  have hp := ready.grid p
  rw [keyPointer,keyABI_modular _ _ _ _ _ _ _ _ _ _ _ _ hs hl hp.2.1 (dim_bound c d hd) ready.group]
  rcases hr with rfl | rfl <;> rfl

theorem query_word (c : Host.Call) (storage : Storage) (buffers : Buffers)
    (ready : Ready c storage buffers) (r : DataRole) (hr : r = .query ∨ r = .output)
    (p : Point c) (lane d : Nat) (hl : lane < 128) (hd : d < c.layout.D)
    (active : p.block*128+lane < c.length p.batch) :
    WordAt (buffers.data r) (dataBytes storage)
      (elementOffset c r p.batch p.head (p.block*128+lane) d)
      (queryPointer c storage buffers r p lane d) := by
  rw [query_pointer_modular c storage buffers ready r hr p lane d hl hd]
  exact complete_word _ _ _ _ (dataBytes_pos storage) (ready.backing r)
    (ready.coordinates r p.batch p.head _ d p.batch.isLt p.head.isLt active hd)

theorem key_word (c : Host.Call) (storage : Storage) (buffers : Buffers)
    (ready : Ready c storage buffers) (r : DataRole) (hr : r = .key ∨ r = .value)
    (p : Point c) (step lane d : Nat) (hl : lane < 128) (hd : d < c.layout.D)
    (active : step*128+lane < c.length p.batch) :
    WordAt (buffers.data r) (dataBytes storage)
      (elementOffset c r p.batch p.head (step*128+lane) d)
      (keyPointer c storage buffers r p step lane d) := by
  rw [key_pointer_modular c storage buffers ready r hr p step lane d hl hd]
  exact complete_word _ _ _ _ (dataBytes_pos storage) (ready.backing r)
    (ready.coordinates r p.batch p.head _ d p.batch.isLt p.head.isLt active hd)

theorem metadata_word_at (c : Host.Call) (storage : Storage) (buffers : Buffers)
    (ready : Ready c storage buffers) (r : MetaRole) (p : Point c) :
    WordAt (buffers.metadataBuffer r).toBuffer (buffers.metadataBuffer r).width.bytes p.batch
      (metaPointer buffers r p.batch) := by
  rw [metaPointer,fromI32_word p.batch (ready.grid p).1,addptr_modular]
  exact complete_word _ _ _ _ (by cases (buffers.metadataBuffer r).width <;> decide)
    (ready.metadata_backing r) (lt_of_lt_of_le p.batch.isLt (ready.metadata_extent r))

end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
