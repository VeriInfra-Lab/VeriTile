import bench.repairs.tritonbench_g.matmul_triton1.revision001.Kernels
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.GroupedGrid
import Mathlib.Data.BitVec

/-! Shared finite arithmetic for the two concrete contiguous matrix repairs.
Both kernels widen every runtime scalar, program id and arange lane to uint64.
Only actually accessed addresses require allocation bounds; padded pointers
retain modular arithmetic. -/
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex
open VeriTile.Triton
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
abbrev Parameters := Core.Parameters
abbrev word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

def ceil (n b : Nat) := (n+b-1)/b
def blocksM (p : Parameters) := ceil p.rows p.blockRows
def blocksN (p : Parameters) := ceil p.columns p.blockColumns
def count (p : Parameters) := blocksM p * blocksN p

def scalarABI (n : Nat) : BitVec 64 :=
  if n = 1 then word 1 else if n < 2^31 then
    (BitVec.ofNat 32 n).signExtend 64 else word n

theorem word_exact {n : Nat} (h : n < 2^64) : (word n).toNat = n := by
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt h]

theorem signed32_cast (n : Nat) (h : n < 2^31) :
    (BitVec.ofNat 32 n).signExtend 64 = word n := by
  have hn : (BitVec.ofNat 32 n).toNat = n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega : n<2^32)]
  change BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt = word n
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]
  exact BitVec.ofInt_natCast 64 n

theorem scalarABI_word (n : Nat) : scalarABI n = word n := by
  unfold scalarABI
  split
  · simp_all
  · split
    · exact signed32_cast n ‹_›
    · rfl

theorem div_word (a b : Nat) (ha : a < 2^64) (hb : b < 2^64) :
    word a / word b = word (a/b) := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_udiv,word_exact ha,word_exact hb]
  exact (word_exact (lt_of_le_of_lt (Nat.div_le_self a b) ha)).symm

theorem mod_word (a b : Nat) (ha : a < 2^64) (hb : b < 2^64) :
    word a % word b = word (a%b) := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_umod,word_exact ha,word_exact hb]
  exact (word_exact (lt_of_le_of_lt (Nat.mod_le a b) ha)).symm

theorem sub_word (a b : Nat) (h : b ≤ a) : word a - word b = word (a-b) := by
  have he : a=(a-b)+b := by omega
  conv_lhs => rw [he]
  change BitVec.ofNat 64 (a-b+b) - BitVec.ofNat 64 b = BitVec.ofNat 64 (a-b)
  rw [BitVec.ofNat_add]
  simp

theorem lt_word (a b : Nat) (ha : a < 2^64) (hb : b < 2^64) :
    (word a).ult (word b) = Bool.true ↔ a < b := by
  rw [BitVec.ult_iff_toNat_lt,word_exact ha,word_exact hb]

def cdiv (n b : BitVec 64) : BitVec 64 := (n+b-word 1)/b

theorem cdiv_word (n b : Nat) (hn : n < 2^63) (hb : 0 < b ∧ b ≤ 256) :
    cdiv (word n) (word b) = word (ceil n b) := by
  unfold cdiv ceil
  rw [←BitVec.ofNat_add,sub_word (n+b) 1 (by omega)]
  exact div_word _ _ (by omega) (by omega)

/-- Ordinary successful launch and checked dimensions; no 32-bit extent cap.
The two sources use fixed small blocks and group size one or eight. -/
structure Ready (p : Parameters) : Prop where
  rows : p.rows < 2^63
  columns : p.columns < 2^63
  inner : p.innerSize < 2^63
  blockRows : 0 < p.blockRows ∧ p.blockRows ≤ 256
  blockColumns : 0 < p.blockColumns ∧ p.blockColumns ≤ 256
  blockInner : 1 < p.blockInner ∧ p.blockInner ≤ 256
  group : 0 < p.groupRows ∧ p.groupRows ≤ 8
  launch : count p < 2^31

structure ActiveProgram (p : Parameters) (pid : Nat) : Prop where
  active : pid < count p

theorem block_limits (p : Parameters) (h : Ready p) (pid : Nat) (hp : pid < count p) :
    0 < blocksM p ∧ 0 < blocksN p ∧ blocksM p < 2^31 ∧ blocksN p < 2^31 := by
  have positive : 0 < blocksM p * blocksN p := by change pid < blocksM p * blocksN p at hp; omega
  have hm : 0 < blocksM p := Nat.pos_of_mul_pos_right positive
  have hn : 0 < blocksN p := Nat.pos_of_mul_pos_left positive
  have ml := Nat.le_mul_of_pos_right (blocksM p) hn
  have nl := Nat.le_mul_of_pos_left (blocksN p) hm
  have small := h.launch
  unfold count at small
  omega

def rowBlock (p : Parameters) (grouped : Bool) (pid : BitVec 64) : BitVec 64 :=
  let m := cdiv (word p.rows) (word p.blockRows)
  let n := cdiv (word p.columns) (word p.blockColumns)
  if grouped then
    let first := pid/(word p.groupRows*n)*word p.groupRows
    let size := if (m-first).ult (word p.groupRows) then m-first else word p.groupRows
    first+pid%size
  else pid/n

def columnBlock (p : Parameters) (grouped : Bool) (pid : BitVec 64) : BitVec 64 :=
  let m := cdiv (word p.rows) (word p.blockRows)
  let n := cdiv (word p.columns) (word p.blockColumns)
  if grouped then
    let first := pid/(word p.groupRows*n)*word p.groupRows
    let size := if (m-first).ult (word p.groupRows) then m-first else word p.groupRows
    (pid%(word p.groupRows*n))/size
  else pid%n

def naturalRow (p : Parameters) (grouped : Bool) (pid : Nat) : Nat :=
  if grouped then GroupedGrid.row (blocksM p) (blocksN p) p.groupRows pid else pid/blocksN p

def naturalColumn (p : Parameters) (grouped : Bool) (pid : Nat) : Nat :=
  if grouped then GroupedGrid.col (blocksM p) (blocksN p) p.groupRows pid else pid%blocksN p

theorem group_size_word (m n g pid : Nat)
    (hm : m < 2^31) (hn : 0 < n ∧ n < 2^31) (hg : 0 < g ∧ g ≤ 8)
    (hp : pid < m*n) (hpid : pid < 2^31) :
    (if (word m - word (GroupedGrid.first g n pid)).ult (word g) then
      word m - word (GroupedGrid.first g n pid) else word g) =
        word (GroupedGrid.size m n g pid) := by
  have hf := (GroupedGrid.facts m n g pid hn.1 hg.1 hp).1
  rw [sub_word m _ (by omega)]
  by_cases small : m-GroupedGrid.first g n pid < g
  · rw [if_pos ((lt_word _ _ (by omega) (by omega)).mpr small)]
    simp [GroupedGrid.size,Nat.min_eq_left (by omega : m-GroupedGrid.first g n pid ≤ g)]
  · rw [if_neg ((lt_word _ _ (by omega) (by omega)).not.mpr small)]
    simp [GroupedGrid.size,Nat.min_eq_right (by omega : g ≤ m-GroupedGrid.first g n pid)]

theorem block_coordinates (p : Parameters) (h : Ready p) (grouped : Bool)
    (pid : Nat) (hp : pid < count p) :
    rowBlock p grouped (word pid) = word (naturalRow p grouped pid) ∧
    columnBlock p grouped (word pid) = word (naturalColumn p grouped pid) := by
  obtain ⟨hm,hn,hmSmall,hnSmall⟩ := block_limits p h pid hp
  have pidSmall : pid < 2^31 := lt_trans hp h.launch
  have groupSmall : p.groupRows*blocksN p < 2^64 := by have := h.group.2; nlinarith
  have first : word pid / (word p.groupRows * word (blocksN p)) * word p.groupRows =
      word (GroupedGrid.first p.groupRows (blocksN p) pid) := by
    rw [←BitVec.ofNat_mul,div_word _ _ (by omega) groupSmall,←BitVec.ofNat_mul]
    rfl
  have hf := (GroupedGrid.facts (blocksM p) (blocksN p) p.groupRows pid hn h.group.1 hp).1
  have hs : GroupedGrid.size (blocksM p) (blocksN p) p.groupRows pid < 2^64 := by
    have := min_le_right (blocksM p-GroupedGrid.first p.groupRows (blocksN p) pid) p.groupRows
    change GroupedGrid.size (blocksM p) (blocksN p) p.groupRows pid ≤ p.groupRows at this
    have := h.group.2
    omega
  cases grouped
  · simp only [rowBlock,columnBlock,naturalRow,naturalColumn,Bool.false_eq_true,↓reduceIte,
      cdiv_word _ _ h.rows h.blockRows,cdiv_word _ _ h.columns h.blockColumns]
    exact ⟨div_word _ _ (by omega) (by exact lt_trans hnSmall (by norm_num)),
      mod_word _ _ (by omega) (by exact lt_trans hnSmall (by norm_num))⟩
  · simp only [rowBlock,columnBlock,naturalRow,naturalColumn,↓reduceIte,
      cdiv_word _ _ h.rows h.blockRows,cdiv_word _ _ h.columns h.blockColumns]
    have sz := group_size_word _ _ _ _ hmSmall ⟨hn,hnSmall⟩ h.group hp pidSmall
    simp only [blocksM,blocksN] at first sz
    simp only [first,sz]

    constructor
    · change word (GroupedGrid.first p.groupRows (blocksN p) pid) +
        word pid % word (GroupedGrid.size (blocksM p) (blocksN p) p.groupRows pid) = _
      rw [mod_word _ _ (by omega) hs,←BitVec.ofNat_add]
      rfl
    · change (word pid % (word p.groupRows*word (blocksN p))) /
        word (GroupedGrid.size (blocksM p) (blocksN p) p.groupRows pid) = _
      rw [←BitVec.ofNat_mul,mod_word _ _ (by omega) groupSmall,
        div_word _ _ (lt_of_le_of_lt (Nat.mod_le _ _) (by omega)) hs]
      rfl


theorem natural_bounds (p : Parameters) (h : Ready p) (grouped : Bool)
    (pid : Nat) (hp : pid < count p) :
    naturalRow p grouped pid < blocksM p ∧ naturalColumn p grouped pid < blocksN p := by
  obtain ⟨hm,hn,_,_⟩ := block_limits p h pid hp
  cases grouped
  · exact ⟨(Nat.div_lt_iff_lt_mul hn).mpr hp,Nat.mod_lt _ hn⟩
  · exact GroupedGrid.bounds _ _ _ _ hn h.group.1 hp

def row (p : Parameters) (grouped : Bool) (pid lane : Nat) : BitVec 64 :=
  rowBlock p grouped (word pid) * word p.blockRows + word lane

def column (p : Parameters) (grouped : Bool) (pid lane : Nat) : BitVec 64 :=
  columnBlock p grouped (word pid) * word p.blockColumns + word lane

theorem padded_small (tiles tile index lane : Nat) (ht : tiles < 2^31)
    (hb : 0 < tile ∧ tile ≤ 256) (hi : index < tiles) (hl : lane < tile) :
    index*tile+lane < 2^64 := by
  have a := Nat.mul_le_mul_right tile (show index+1 ≤ tiles by omega)
  have b := Nat.mul_le_mul_right 256 (show tiles ≤ 2^31 by omega)
  have c := Nat.mul_le_mul_left tiles hb.2
  nlinarith

theorem coordinates (p : Parameters) (h : Ready p) (grouped : Bool)
    (pid i j : Nat) (hp : pid < count p) (hi : i < p.blockRows) (hj : j < p.blockColumns) :
    (row p grouped pid i).toNat = naturalRow p grouped pid*p.blockRows+i ∧
    (column p grouped pid j).toNat = naturalColumn p grouped pid*p.blockColumns+j := by
  obtain ⟨hr,hc⟩ := block_coordinates p h grouped pid hp
  obtain ⟨hm,hn,ms,ns⟩ := block_limits p h pid hp
  obtain ⟨rb,cb⟩ := natural_bounds p h grouped pid hp
  simp only [row,column,hr,hc,word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  exact ⟨word_exact (padded_small _ _ _ _ ms h.blockRows rb hi),
    word_exact (padded_small _ _ _ _ ns h.blockColumns cb hj)⟩

theorem loop_body (n b k : Nat) (hb : 0 < b) (hk : k < ceil n b) : k*b < n := by
  have h := Nat.mul_le_mul_right b (show k+1 ≤ ceil n b by omega)
  have bound := Nat.div_mul_le_self (n+b-1) b
  unfold ceil at h
  rw [Nat.add_mul,Nat.one_mul] at h
  omega

theorem loop_padded_small (p : Parameters) (h : Ready p) (k lane : Nat)
    (hk : k ≤ ceil p.innerSize p.blockInner) (hl : lane ≤ p.blockInner) :
    k*p.blockInner+lane < 2^64 := by
  have a := Nat.mul_le_mul_right p.blockInner hk
  have b := Nat.div_mul_le_self (p.innerSize+p.blockInner-1) p.blockInner
  unfold ceil at a
  have := h.inner
  have := h.blockInner.2
  omega

theorem loop_bound_small (p : Parameters) (h : Ready p) :
    ceil p.innerSize p.blockInner < 2^63 := by
  have hb := Nat.div_mul_le_self (p.innerSize+p.blockInner-1) p.blockInner
  change ceil p.innerSize p.blockInner*p.blockInner ≤ _ at hb
  have prod := Nat.mul_le_mul_left (ceil p.innerSize p.blockInner)
    (show 2 ≤ p.blockInner from h.blockInner.1)
  have := h.inner
  have := h.blockInner.2
  omega

theorem loop_bound (p : Parameters) (h : Ready p) :
    (cdiv (word p.innerSize) (word p.blockInner)).toNat = ceil p.innerSize p.blockInner ∧
    (cdiv (word p.innerSize) (word p.blockInner)).toInt = (ceil p.innerSize p.blockInner : Int) := by
  rw [cdiv_word _ _ h.inner ⟨by have := h.blockInner.1; omega,h.blockInner.2⟩]
  have hs := loop_bound_small p h
  have hn := word_exact (show ceil p.innerSize p.blockInner < 2^64 by omega)
  exact ⟨hn,by rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]⟩

theorem loop_test (p : Parameters) (h : Ready p) (k : Nat)
    (hk : k ≤ ceil p.innerSize p.blockInner) :
    (word k).ult (cdiv (word p.innerSize) (word p.blockInner)) = Bool.true ↔
      k < ceil p.innerSize p.blockInner := by
  rw [BitVec.ult_iff_toNat_lt,(loop_bound p h).1,
    word_exact (by have := loop_bound_small p h; omega)]

theorem remaining (p : Parameters) (h : Ready p) (k : Nat)
    (hk : k < ceil p.innerSize p.blockInner) :
    word p.innerSize - word k*word p.blockInner = word (p.innerSize-k*p.blockInner) := by
  rw [←BitVec.ofNat_mul]
  exact sub_word _ _ (by have := loop_body _ _ _ (by have := h.blockInner.1; omega) hk; omega)

theorem masks (p : Parameters) (h : Ready p) (grouped : Bool)
    (pid i j k e : Nat) (hp : pid < count p) (hi : i < p.blockRows)
    (hj : j < p.blockColumns) (hk : k < ceil p.innerSize p.blockInner) (he : e < p.blockInner) :
    ((row p grouped pid i).ult (word p.rows) = Bool.true ↔ naturalRow p grouped pid*p.blockRows+i < p.rows) ∧
    ((column p grouped pid j).ult (word p.columns) = Bool.true ↔ naturalColumn p grouped pid*p.blockColumns+j < p.columns) ∧
    ((word e).ult (word p.innerSize-word k*word p.blockInner) = Bool.true ↔ e < p.innerSize-k*p.blockInner) := by
  obtain ⟨hr,hc⟩ := coordinates p h grouped pid i j hp hi hj
  have er : p.rows < 2^64 := by have := h.rows; omega
  have ec : p.columns < 2^64 := by have := h.columns; omega
  have ee : e < 2^64 := by have := h.blockInner.2; omega
  have ek : p.innerSize-k*p.blockInner < 2^64 := by have := h.inner; omega
  simp only [BitVec.ult_iff_toNat_lt,hr,hc,word_exact er,word_exact ec,
    remaining p h k hk,word_exact ee,word_exact ek,true_and]

def aOffset (p : Parameters) (grouped : Bool) (pid i e k : Nat) : BitVec 64 :=
  row p grouped pid i*word p.innerSize + word e + word k*word p.blockInner

def bOffset (p : Parameters) (grouped : Bool) (pid j e k : Nat) : BitVec 64 :=
  word e*word p.columns+column p grouped pid j + word k*(word p.blockInner*word p.columns)

def cOffset (p : Parameters) (grouped : Bool) (pid i j : Nat) : BitVec 64 :=
  row p grouped pid i*word p.columns+column p grouped pid j

theorem pointer_steps (p : Parameters) (grouped : Bool) (pid i j e k : Nat) :
    aOffset p grouped pid i e (k+1) = aOffset p grouped pid i e k+word p.blockInner ∧
    bOffset p grouped pid j e (k+1) = bOffset p grouped pid j e k+word p.blockInner*word p.columns := by
  simp only [aOffset,bOffset,word,BitVec.ofNat_add,BitVec.ofNat_one,add_mul,one_mul]
  constructor <;> ac_rfl

/-- `addptr` sign interpretation and byte scaling preserve the same 64-bit
residue even for inactive padded addresses. -/
theorem signed_pointer_offset (offset : BitVec 64) :
    BitVec.ofInt 64 (4*offset.toInt) = word 4*offset := by
  rw [BitVec.ofInt_mul,BitVec.ofInt_toInt]
  rfl

theorem linear_bound (r c rows cols : Nat) (hr : r < rows) (hc : c < cols) :
    r*cols+c < rows*cols := by
  have := Nat.mul_le_mul_right cols (show r+1 ≤ rows by omega)
  nlinarith

structure StorageBounds (p : Parameters) : Prop where
  input : p.rows*p.innerSize*4 ≤ 2^64
  weight : p.innerSize*p.columns*4 ≤ 2^64
  output : p.rows*p.columns*4 ≤ 2^64

theorem offsets (p : Parameters) (h : Ready p) (grouped : Bool)
    (pid i j e k : Nat) (hp : pid < count p) :
    aOffset p grouped pid i e k = word ((naturalRow p grouped pid*p.blockRows+i)*p.innerSize+(k*p.blockInner+e)) ∧
    bOffset p grouped pid j e k = word ((k*p.blockInner+e)*p.columns+(naturalColumn p grouped pid*p.blockColumns+j)) ∧
    cOffset p grouped pid i j = word ((naturalRow p grouped pid*p.blockRows+i)*p.columns+(naturalColumn p grouped pid*p.blockColumns+j)) := by
  obtain ⟨hr,hc⟩ := block_coordinates p h grouped pid hp
  simp only [aOffset,bOffset,cOffset,row,column,hr,hc,word,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  constructor
  · congr 1; ring
  constructor
  · congr 1; ring
  · trivial

theorem accessed_bytes (offset count base : Nat)
    (ho : offset < count) (hb : base+count*4 ≤ 2^64) :
    (word base+word offset*word 4).toNat = base+offset*4 ∧
    base+offset*4+3 < base+count*4 := by
  have bound : base+offset*4+3 < base+count*4 := by omega
  constructor
  · simp only [word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
    exact word_exact (by omega)
  · exact bound


theorem a_active (p : Parameters) (h : Ready p) (storage : StorageBounds p)
    (grouped : Bool) (pid i e k : Nat) (hp : pid < count p)
    (hr : naturalRow p grouped pid*p.blockRows+i < p.rows)
    (he : e < p.innerSize-k*p.blockInner) :
    (aOffset p grouped pid i e k).toNat =
      (naturalRow p grouped pid*p.blockRows+i)*p.innerSize+(k*p.blockInner+e) := by
  rw [(offsets p h grouped pid i 0 e k hp).1]
  have bound := linear_bound _ _ _ _ hr (by omega : k*p.blockInner+e < p.innerSize)
  have := storage.input
  exact word_exact (by omega)

theorem b_active (p : Parameters) (h : Ready p) (storage : StorageBounds p)
    (grouped : Bool) (pid j e k : Nat) (hp : pid < count p)
    (hc : naturalColumn p grouped pid*p.blockColumns+j < p.columns)
    (he : e < p.innerSize-k*p.blockInner) :
    (bOffset p grouped pid j e k).toNat =
      (k*p.blockInner+e)*p.columns+(naturalColumn p grouped pid*p.blockColumns+j) := by
  rw [(offsets p h grouped pid 0 j e k hp).2.1]
  have bound := linear_bound _ _ _ _ (by omega : k*p.blockInner+e < p.innerSize) hc
  have := storage.weight
  exact word_exact (by omega)

theorem c_active (p : Parameters) (h : Ready p) (storage : StorageBounds p)
    (grouped : Bool) (pid i j : Nat) (hp : pid < count p)
    (hr : naturalRow p grouped pid*p.blockRows+i < p.rows)
    (hc : naturalColumn p grouped pid*p.blockColumns+j < p.columns) :
    (cOffset p grouped pid i j).toNat =
      (naturalRow p grouped pid*p.blockRows+i)*p.columns+(naturalColumn p grouped pid*p.blockColumns+j) := by
  rw [(offsets p h grouped pid i j 0 0 hp).2.2]
  have bound := linear_bound _ _ _ _ hr hc
  have := storage.output
  exact word_exact (by omega)

#axiomsClean scalarABI_word
#axiomsClean cdiv_word
#axiomsClean block_coordinates
#axiomsClean coordinates
#axiomsClean loop_bound
#axiomsClean loop_test
#axiomsClean masks
#axiomsClean pointer_steps
#axiomsClean offsets
#axiomsClean accessed_bytes
#axiomsClean a_active
#axiomsClean b_active
#axiomsClean c_active
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex
