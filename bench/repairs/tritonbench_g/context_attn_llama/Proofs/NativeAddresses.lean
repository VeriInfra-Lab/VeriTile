import bench.repairs.tritonbench_g.context_attn_llama.Proofs.NativeControl
import bench.tritonbench_g.context_attn_llama.Proofs.Host

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 3000000

structure Widths where
  start : MetadataWidth
  sequence : MetadataWidth
  prompt : MetadataWidth
  request : MetadataWidth
  pages : MetadataWidth

/-- Byte placement of the resolved read buffers and fresh output allocation.
The extent is the tensor allocation extent, including its storage offset. -/
structure Bytes (A : FlatAlloc) where
  base : RegionName → Nat
  size : RegionName → Nat
  positive : ∀ rg, 0 < size rg
  backing : ∀ rg, base rg + size rg * A.extent rg ≤ 2^64

/-- Actual scalar/grid/metadata representation and byte allocation conditions.
These are descriptor facts, not bounds on inactive address intermediates. -/
structure ABI (c : Host.Call) (A : FlatAlloc) (w : Widths) (B : Bytes A) : Prop where
  grid0 : c.blocks ≤ 2^31
  grid1 : c.batches*c.layout.heads ≤ 2^31
  heads : c.layout.heads < 2^63
  group : c.layout.kvGroup < 2^63
  strides : ∀ n, n ∈ [c.layout.qRow,c.layout.qHead,c.layout.qDim,
    c.layout.kRow,c.layout.kHead,c.layout.kDim,c.layout.vRow,c.layout.vHead,c.layout.vDim,
    c.layout.outRow,c.layout.outHead,c.layout.outDim,c.layout.reqRow,c.layout.reqToken] → n < 2^63
  start : ∀ b, b < c.batches → w.start.Fits (c.start b)
  sequence : ∀ b, b < c.batches → w.sequence.Fits (c.length b+c.prompt b)
  prompt : ∀ b, b < c.batches → w.prompt.Fits (c.prompt b)
  request : ∀ b, b < c.batches → w.request.Fits (c.request b)
  pages : ∀ b, b < c.batches → ∀ j, j < c.length b+c.prompt b → w.pages.Fits (c.page b j)
  data_bytes : ∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,c.regions.out] → B.size rg = 4
  start_bytes : B.size c.regions.start = w.start.bytes
  sequence_bytes : B.size c.regions.sequence = w.sequence.bytes
  prompt_bytes : B.size c.regions.prompt = w.prompt.bytes
  request_bytes : B.size c.regions.request = w.request.bytes
  pages_bytes : B.size c.regions.pages = w.pages.bytes
  fresh : ∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,c.regions.start,
      c.regions.sequence,c.regions.prompt,c.regions.request,c.regions.pages] →
    B.base c.regions.out+B.size c.regions.out*A.extent c.regions.out ≤ B.base rg ∨
      B.base rg+B.size rg*A.extent rg ≤ B.base c.regions.out

def batchBits (p : Layout) (l : Formula.Launch) := (fromI32 l.batchHead).sdiv (scalar p.heads)
def headBits (p : Layout) (l : Formula.Launch) := (fromI32 l.batchHead).srem (scalar p.heads)
def kvBits (p : Layout) (l : Formula.Launch) := (headBits p l).sdiv (scalar p.kvGroup)

theorem launch_exact (p : Layout) (l : Formula.Launch)
    (hbh : l.batchHead < 2^31) (hh : p.heads < 2^63) (hg : p.kvGroup < 2^63) :
    batchBits p l = word (Formula.batch p l) ∧ headBits p l = word (Formula.head p l) ∧
    kvBits p l = word (Formula.head p l / p.kvGroup) := by
  have hb : l.batchHead < 2^63 := by omega
  have hr : l.batchHead % p.heads < 2^63 := lt_of_le_of_lt (Nat.mod_le _ _) hb
  simp only [batchBits,headBits,kvBits,fromI32_eq l.batchHead hbh,scalar_eq,
    sdiv_eq _ _ hb hh,srem_eq _ _ hb hh,sdiv_eq _ _ hr hg,Formula.batch,Formula.head,and_self]

def queryBits (p : Layout) (l : Formula.Launch) (w : MetadataWidth) (start i d : Nat)
    (sr sh sd : Nat) :=
  (metadata w start+rowBits l.block p.BM i)*scalar sr+
    headBits p l*scalar sh+fromI32 d*scalar sd

def pageOffsetBits (p : Layout) (l : Formula.Launch) (w : MetadataWidth)
    (req step j : Nat) := scalar p.reqRow*metadata w req+scalar p.reqToken*keyBits step p.BN j

def valueBits (p : Layout) (l : Formula.Launch) (page : BitVec 64) (d sr sh sd : Nat) :=
  page*scalar sr+kvBits p l*scalar sh+fromI32 d*scalar sd

theorem query_modular (p : Layout) (l : Formula.Launch) (w : MetadataWidth) (start i d sr sh sd : Nat)
    (hbl : l.block < 2^31) (hbh : l.batchHead < 2^31) (hm : Tile p.BM)
    (hh : p.heads < 2^63) (hg : p.kvGroup < 2^63) (hs : w.Fits start)
    (hi : i < p.BM) (hd : d < 2^31) :
    queryBits p l w start i d sr sh sd =
      word ((start+Formula.queryRow p l i)*sr+Formula.head p l*sh+d*sd) := by
  rw [queryBits,metadata_eq w start hs,row_modular _ _ _ hbl hm hi,
    (launch_exact p l hbh hh hg).2.1,fromI32_eq d hd]
  simp only [scalar_eq,word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat,Formula.queryRow]

theorem page_modular (p : Layout) (l : Formula.Launch) (w : MetadataWidth) (req step j : Nat)
    (hm : Tile p.BN) (hr : w.Fits req) (hj : j < p.BN) :
    pageOffsetBits p l w req step j = word (p.reqRow*req+p.reqToken*(step*p.BN+j)) := by
  rw [pageOffsetBits,metadata_eq w req hr,key_modular step p.BN j hm hj]
  simp only [scalar_eq,word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]

theorem value_modular (p : Layout) (l : Formula.Launch) (pg d sr sh sd : Nat)
    (hbh : l.batchHead < 2^31) (hh : p.heads < 2^63) (hg : p.kvGroup < 2^63) (hd : d < 2^31) :
    valueBits p l (word pg) d sr sh sd = word (pg*sr+(Formula.head p l/p.kvGroup)*sh+d*sd) := by
  rw [valueBits,(launch_exact p l hbh hh hg).2.2,fromI32_eq d hd]
  simp only [scalar_eq,word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]

theorem block_abi (c : Host.Call) (A : FlatAlloc) (w : Widths) (B : Bytes A)
    (abi : ABI c A w B) (idx : GridIndex c.grid) : (c.launchAt idx).block < 2^31 :=
  lt_of_lt_of_le (Host.block_bound c idx) abi.grid0

theorem batchHead_abi (c : Host.Call) (A : FlatAlloc) (w : Widths) (B : Bytes A)
    (abi : ABI c A w B) (idx : GridIndex c.grid) : (c.launchAt idx).batchHead < 2^31 := by
  have h : (c.launchAt idx).batchHead < c.batches*c.layout.heads := (idx ⟨1,by simp [Host.Call.grid,Grid.rank]⟩).isLt
  exact lt_of_lt_of_le h abi.grid1

theorem dim_abi (c : Host.Call) (d : Nat) (hd : d < c.layout.D) : d < 2^31 := by
  have h := c.channels
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h <;> omega

/-- The row mask also covers padded lanes and wholly inactive programs. -/
theorem query_mask (ws wp : MetadataWidth) (block bm seq prompt i : Nat)
    (hb : block < 2^31) (hm : Tile bm) (hs : ws.Fits seq) (hp : wp.Fits prompt)
    (hps : prompt ≤ seq) (hi : i < bm) :
    (rowBits block bm i).slt (lengthBits ws wp seq prompt) = Bool.true ↔ block*bm+i < seq-prompt := by
  have hs63 := metadata_lt ws seq hs
  rw [row_modular block bm i hb hm hi,length_exact ws wp seq prompt hs hp hps]
  apply slt_exact
  · rcases hm with rfl|rfl <;> omega
  · omega

/-- Signed offset interpretation in addptr has the correct byte residue even
if the element offset has wrapped. Only a true mask demands a backing cell. -/
theorem masked_pointer (A : FlatAlloc) (B : Bytes A) (rg : RegionName) (on : Bool)
    (off : BitVec 64) (math : Nat) (he : off = word math)
    (hb : on = Bool.true → math < A.extent rg) :
    access on (B.base rg) (B.size rg) off =
      if on then some (B.base rg+B.size rg*math) else none := by
  rw [he]
  exact access_exact on _ _ _ _ (B.positive rg) (B.backing rg) hb

end
end VeriTile.Bench.Repairs.ContextAttnLlama.Native
