import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.HostPlan
import bench.tritonbench_g.context_attn_nopad.Proofs.PreludeExecution

namespace VeriTile.Bench.Repairs.ContextAttnNopad.NativeProgram
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Formula Program Host HostPlan
noncomputable section
set_option maxHeartbeats 2000000

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

def pidWord {n limit : Nat} (j : Axis n limit) : Word := word j.2.val + word j.1.val.val

theorem pidWord_eq {n limit : Nat} (j : Axis n limit) : pidWord j = word (global j).val := by
  simp only [pidWord, global, word, ← BitVec.ofNat_add]
  congr 1
  omega

/-- Intrinsic runtime representation facts about successful ordinary tensor
objects/allocations. This is not a new safe-shape condition on the kernel:
Torch's shape and allocation extent ABI is signed 64-bit. -/
structure ABI (c : Call) (A : FlatAlloc) : Prop where
  heads : c.q.heads ≤ 2^63
  bytes : ∀ rg, 4*A.extent rg ≤ 2^63

theorem ABI.extent {c : Call} {A : FlatAlloc} (ha : ABI c A) (rg : RegionName) :
    A.extent rg ≤ 2^63 := by
  have := ha.bytes rg
  omega

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

theorem effective_blocks_bound {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) : (effective c r s).blocks ≤ 16777216 := by
  have hm' := maximum_lt_int32 hm.shape
  rw [← metadataMaximum_eq hm] at hm'
  change (min c.maxInputLen (metadataMaximum c r s)+127)/128 ≤ _
  norm_num at hm'
  omega

/-- Every actual native word used by the source control flow or an active
memory operation projects to the corresponding mathematical integer. Masked
pointer formation is allowed to wrap; it is never converted into an active
access assumption. -/
structure Projects (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (j : Job c) : Prop where
  byteOffset : ∀ rg addr, addr < A.extent rg →
    (word addr * word 4).toInt = ((addr*4 : Nat) : Int)
  batch : (pidWord j.1).toInt = ((global j.1).val : Int)
  head : (pidWord j.2.1).toInt = ((global j.2.1).val : Int)
  block : (pidWord j.2.2).toInt = ((global j.2.2).val : Int)
  start : metadataWord ((inputs s r).start (global j.1).val) = word (c.start (global j.1).val)
  length : metadataWord ((inputs s r).sequence (global j.1).val) = word (c.length (global j.1).val)
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
    (hm : Memory c r A s) (ha : ABI c A) (hb : c.blocks ≤ 16777216) (j : Job c) :
    Projects c r A s j := by
  have hbatch := (global j.1).isLt
  have hhead := (global j.2.1).isLt
  have hblock := (global j.2.2).isLt
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
  · rw [hm.start _ hbatch]; exact metadataWord_exact (hm.shape.int32Start _ hbatch)
  · rw [hm.length _ hbatch]; exact metadataWord_exact (hm.shape.int32Length _ hbatch)
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

/-- Source PID prefix evaluated using actual signed-64 words. The unchanged
body consumes their mathematical values after the checked projection above. -/
def nativePids (c : Call) (j : Job c) : Nat → Nat
  | 0 => (pidWord j.1).toInt.toNat
  | 1 => (pidWord j.2.1).toInt.toNat
  | 2 => (pidWord j.2.2).toInt.toNat
  | _ => 0

theorem nativePids_eq {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    {j : Job c} (hp : Projects c r A s j) : nativePids c j = (index c j).toPids := by
  funext ax
  rcases ax with _ | (_ | (_ | ax))
  · simp [nativePids, hp.batch, index, GridIndex.toPids, Call.grid, Grid.rank, Call.index]
  · simp [nativePids, hp.head, index, GridIndex.toPids, Call.grid, Grid.rank, Call.index]
  · simp [nativePids, hp.block, index, GridIndex.toPids, Call.grid, Grid.rank, Call.index]
    rfl
  · simp [nativePids, index, GridIndex.toPids, Call.grid, Grid.rank, Call.index]

/-- Projected source state. numPids is logical-grid bookkeeping: this kernel
never reads tl.num_programs (also checked by the complete AST receipt). The
actual clipped native dimensions are separately recorded by nativeGrid. -/
def atProgram (c : Call) (j : Job c) (s : BlockState) : BlockState :=
  (s.withPids (nativePids c j)).withNumPids c.grid.toNumPids

theorem atProgram_eq {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    {j : Job c} (hp : Projects c r A s j) (u : BlockState) :
    atProgram c j u = u.withGridIndex (index c j) := by
  rw [atProgram, nativePids_eq hp]
  rfl

/-- Intrinsic signed-64 tensor shape representation, independent of values. -/
structure ShapeABI (c : Call) : Prop where
  batches : c.batches ≤ 2^63
  heads : c.q.heads ≤ 2^63

theorem shapeABI {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (ha : ABI c A) : ShapeABI c :=
  ⟨le_trans hm.startExtent (ha.extent r.start),ha.heads⟩

theorem nativePids_of_shape {c : Call} (ha : ShapeABI c) (hb : c.blocks ≤ 16777216)
    (j : Job c) : nativePids c j = (index c j).toPids := by
  have hbatch : (pidWord j.1).toInt = ((global j.1).val : Int) := by
    rw [pidWord_eq]; exact word_exact (lt_of_lt_of_le (global j.1).isLt ha.batches)
  have hhead : (pidWord j.2.1).toInt = ((global j.2.1).val : Int) := by
    rw [pidWord_eq]; exact word_exact (lt_of_lt_of_le (global j.2.1).isLt ha.heads)
  have hblock : (pidWord j.2.2).toInt = ((global j.2.2).val : Int) := by
    rw [pidWord_eq]; apply word_exact; have := (global j.2.2).isLt; omega
  funext ax
  rcases ax with _ | (_ | (_ | ax))
  · simp [nativePids,hbatch,index,GridIndex.toPids,Call.grid,Grid.rank,Call.index]
  · simp [nativePids,hhead,index,GridIndex.toPids,Call.grid,Grid.rank,Call.index]
  · simp [nativePids,hblock,index,GridIndex.toPids,Call.grid,Grid.rank,Call.index]
    rfl
  · simp [nativePids,index,GridIndex.toPids,Call.grid,Grid.rank,Call.index]

/-- Shared floating body, executed at the candidate's evaluated native PID
prefix. All finite integer address and control sites are exposed by Projects;
no numerical law or successful-execution premise is hidden in this definition. -/
def execute (c : Call) (r : Regions) (A : FlatAlloc) (j : Job c) (s : BlockState) :
    Option BlockState :=
  exec (A.flattenKernel (kernel c r).eraseDType.toAlgKernel) (atProgram c j s)

theorem execute_projects {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    {j : Job c} (hp : Projects c r A s j) (u : BlockState) :
    execute c r A j u =
      exec (A.flattenKernel (c.source r Bool.false).eraseDType.toAlgKernel)
        (u.withGridIndex (index c j)) := by
  rw [execute, atProgram_eq hp]
  rfl

end
end VeriTile.Bench.Repairs.ContextAttnNopad.NativeProgram
