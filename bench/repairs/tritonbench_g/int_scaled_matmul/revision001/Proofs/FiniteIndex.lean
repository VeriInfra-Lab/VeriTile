import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Contract
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.GroupedGrid
import Mathlib.Data.BitVec

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FiniteIndex
open Contract
abbrev word (n : Nat) : BitVec 32 := BitVec.ofNat 32 n

theorem toNat {n : Nat} (h : n < 2^32) : (word n).toNat = n := Nat.mod_eq_of_lt h
theorem toInt {n : Nat} (h : n < 2^31) : (word n).toInt = (n : Int) := by
  have hn := toNat (show n < 2^32 by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]
theorem msb {n : Nat} (h : n < 2^31) : (word n).msb = Bool.false := by
  apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
  rw [toNat (by omega)]
  omega
theorem sdiv (a b : Nat) (ha : a < 2^31) (hb : b < 2^31) :
    (word a).sdiv (word b) = word (a/b) := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_sdiv,msb ha,msb hb]
  change ((word a)/(word b)).toNat = _
  rw [BitVec.toNat_udiv,toNat (by omega),toNat (by omega),toNat (by have := Nat.div_le_self a b; omega)]
theorem srem (a b : Nat) (ha : a < 2^31) (hb : b < 2^31) :
    (word a).srem (word b) = word (a%b) := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_srem,msb ha,msb hb,toNat (by omega),toNat (by omega),
    toNat (by have := Nat.mod_le a b; omega)]
theorem sub (a b : Nat) (h : b ≤ a) : word a-word b = word (a-b) := by
  have he : a=(a-b)+b := by omega
  conv_lhs => rw [he]
  change BitVec.ofNat 32 (a-b+b)-BitVec.ofNat 32 b = BitVec.ofNat 32 (a-b)
  rw [BitVec.ofNat_add]
  simp
theorem slt (a b : Nat) (ha : a < 2^31) (hb : b < 2^31) :
    (word a).slt (word b) = Bool.true ↔ a < b := by
  rw [BitVec.slt_iff_toInt_lt,toInt ha,toInt hb]
  omega

def cdiv (n b : BitVec 32) : BitVec 32 := (n+b-word 1).sdiv b
theorem cdiv_exact (n b : Nat) (hb : 0 < b) (he : n+b-1 < 2^31) (hbmax : b < 2^31) :
    cdiv (word n) (word b) = word (ceil n b) := by
  unfold cdiv ceil
  rw [←BitVec.ofNat_add,sub (n+b) 1 (by omega)]
  exact sdiv _ _ he hbmax

def count (M N : Nat) (p : Config) := ceil M p.blockM*ceil N p.blockN
def rowBlock (M N : Nat) (p : Config) (pid : BitVec 32) : BitVec 32 :=
  let m := cdiv (word M) (word p.blockM)
  let n := cdiv (word N) (word p.blockN)
  let first := (pid.sdiv (word p.groupM*n))*word p.groupM
  let size := if (m-first).slt (word p.groupM) then m-first else word p.groupM
  first+pid.srem size
def columnBlock (M N : Nat) (p : Config) (pid : BitVec 32) : BitVec 32 :=
  let m := cdiv (word M) (word p.blockM)
  let n := cdiv (word N) (word p.blockN)
  let first := (pid.sdiv (word p.groupM*n))*word p.groupM
  let size := if (m-first).slt (word p.groupM) then m-first else word p.groupM
  (pid.srem (word p.groupM*n)).sdiv size

theorem ceil_small (n b : Nat) (hb : 0 < b) (he : n+b-1 < 2^31) :
    n < 2^31 ∧ b ≤ 2^31 ∧ ceil n b < 2^31 ∧ padded n b < 2^31 := by
  have hd := Nat.div_le_self (n+b-1) b
  have hm := Nat.div_mul_le_self (n+b-1) b
  simp only [padded,ceil]
  omega

theorem group_bounds (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (pid : Nat) (hp : pid < count M N p) :
    0 < ceil M p.blockM ∧ 0 < ceil N p.blockN ∧
    ceil M p.blockM < 2^31 ∧ ceil N p.blockN < 2^31 ∧
    0 < p.groupM ∧ p.groupM < 2^31 ∧ p.groupM*ceil N p.blockN < 2^31 ∧ pid < 2^31 := by
  obtain ⟨bm,bn,bk,g⟩ := positive_tiles p legal
  obtain ⟨em,en,ek,ew,ec,ea,eb,eo⟩ := envelope_fields M N K p env
  have pos : 0 < ceil M p.blockM*ceil N p.blockN := by unfold count at hp; omega
  have hm := Nat.pos_of_mul_pos_right pos
  have hn := Nat.pos_of_mul_pos_left pos
  have bound := Nat.le_mul_of_pos_right p.groupM hn
  have ms := (ceil_small M p.blockM bm em).2.2.1
  have ns := (ceil_small N p.blockN bn en).2.2.1
  exact ⟨hm,hn,ms,ns,g,by omega,ew,lt_trans hp ec⟩

theorem group_size (m n g pid : Nat) (hm : m < 2^31) (hn : 0 < n)
    (hg : 0 < g ∧ g < 2^31) (hp : pid < m*n) :
    (if (word m-word (GroupedGrid.first g n pid)).slt (word g) then
      word m-word (GroupedGrid.first g n pid) else word g) =
        word (GroupedGrid.size m n g pid) := by
  have hf := (GroupedGrid.facts m n g pid hn hg.1 hp).1
  rw [sub _ _ (by omega)]
  by_cases h : m-GroupedGrid.first g n pid < g
  · rw [if_pos ((slt _ _ (by omega) hg.2).mpr h)]
    simp [GroupedGrid.size,Nat.min_eq_left (by omega : m-GroupedGrid.first g n pid ≤ g)]
  · rw [if_neg ((slt _ _ (by omega) hg.2).not.mpr h)]
    simp [GroupedGrid.size,Nat.min_eq_right (by omega : g ≤ m-GroupedGrid.first g n pid)]

theorem tiles_small (p : Config) (h : ConfigContract p) :
    p.blockM < 2^31 ∧ p.blockN < 2^31 ∧ p.blockK < 2^31 := by
  obtain ⟨bm,bn,bk,g⟩ := positive_tiles p h
  have products := h.2.2.1
  have a := Nat.le_mul_of_pos_right p.blockM bn
  have b := Nat.le_mul_of_pos_left p.blockN bm
  have k := Nat.le_mul_of_pos_left p.blockK bm
  exact ⟨by omega,by omega,by omega⟩

theorem blocks_exact (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (pid : Nat) (hp : pid < count M N p) :
    rowBlock M N p (word pid) = word (GroupedGrid.row (ceil M p.blockM) (ceil N p.blockN) p.groupM pid) ∧
    columnBlock M N p (word pid) = word (GroupedGrid.col (ceil M p.blockM) (ceil N p.blockN) p.groupM pid) := by
  obtain ⟨bm,bn,bk,g⟩ := positive_tiles p legal
  obtain ⟨em,en,ek,ew,ec,ea,eb,eo⟩ := envelope_fields M N K p env
  obtain ⟨mp,np,ms,ns,gp,gs,ws,ps⟩ := group_bounds M N K p legal env pid hp
  have first : (word pid).sdiv (word p.groupM*word (ceil N p.blockN))*word p.groupM =
      word (GroupedGrid.first p.groupM (ceil N p.blockN) pid) := by
    rw [←BitVec.ofNat_mul,sdiv _ _ ps ws,←BitVec.ofNat_mul]
    rfl
  have size := group_size _ _ _ pid ms np ⟨gp,gs⟩ hp
  have ss : GroupedGrid.size (ceil M p.blockM) (ceil N p.blockN) p.groupM pid < 2^31 :=
    lt_of_le_of_lt (min_le_right _ _) gs
  simp only [rowBlock,columnBlock,cdiv_exact M _ bm em (tiles_small p legal).1,cdiv_exact N _ bn en (tiles_small p legal).2.1,first,size]
  constructor
  · rw [srem _ _ ps ss,←BitVec.ofNat_add]
    rfl
  · rw [←BitVec.ofNat_mul,srem _ _ ps ws,sdiv _ _ (lt_of_le_of_lt (Nat.mod_le _ _) ps) ss]
    rfl

def row (M N : Nat) (p : Config) (pid lane : Nat) := rowBlock M N p (word pid)*word p.blockM+word lane
def column (M N : Nat) (p : Config) (pid lane : Nat) := columnBlock M N p (word pid)*word p.blockN+word lane
def naturalRow (M N : Nat) (p : Config) (pid lane : Nat) :=
  GroupedGrid.row (ceil M p.blockM) (ceil N p.blockN) p.groupM pid*p.blockM+lane
def naturalColumn (M N : Nat) (p : Config) (pid lane : Nat) :=
  GroupedGrid.col (ceil M p.blockM) (ceil N p.blockN) p.groupM pid*p.blockN+lane

theorem coordinate_bounds (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (pid i j : Nat) (hp : pid < count M N p)
    (hi : i < p.blockM) (hj : j < p.blockN) :
    naturalRow M N p pid i < padded M p.blockM ∧ naturalColumn M N p pid j < padded N p.blockN := by
  obtain ⟨_,np,_,_,gp,_,_,_⟩ := group_bounds M N K p legal env pid hp
  obtain ⟨rb,cb⟩ := GroupedGrid.bounds _ _ _ _ np gp hp
  have rm := Nat.mul_le_mul_right p.blockM (show GroupedGrid.row (ceil M p.blockM) (ceil N p.blockN) p.groupM pid+1 ≤ ceil M p.blockM by omega)
  have cn := Nat.mul_le_mul_right p.blockN (show GroupedGrid.col (ceil M p.blockM) (ceil N p.blockN) p.groupM pid+1 ≤ ceil N p.blockN by omega)
  simp only [Nat.add_mul,Nat.one_mul] at rm cn
  unfold naturalRow naturalColumn padded
  omega

theorem coordinates (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (pid i j : Nat) (hp : pid < count M N p)
    (hi : i < p.blockM) (hj : j < p.blockN) :
    row M N p pid i = word (naturalRow M N p pid i) ∧
    column M N p pid j = word (naturalColumn M N p pid j) ∧
    (row M N p pid i).toInt = (naturalRow M N p pid i : Int) ∧
    (column M N p pid j).toInt = (naturalColumn M N p pid j : Int) := by
  obtain ⟨hr,hc⟩ := blocks_exact M N K p legal env pid hp
  obtain ⟨rb,cb⟩ := coordinate_bounds M N K p legal env pid i j hp hi hj
  obtain ⟨bm,bn,_,_⟩ := positive_tiles p legal
  have em := (envelope_fields M N K p env).1
  have en := (envelope_fields M N K p env).2.1
  have ms := (ceil_small M p.blockM bm em).2.2.2
  have ns := (ceil_small N p.blockN bn en).2.2.2
  simp only [row,column,hr,hc,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  exact ⟨rfl,rfl,toInt (lt_trans rb ms),toInt (lt_trans cb ns)⟩

def remaining (K : Nat) (p : Config) (k : Nat) : BitVec 32 := word K-word k*word p.blockK

theorem remaining_exact (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (k : Nat) (hk : k ≤ ceil K p.blockK) :
    (remaining K p k).toInt = (K : Int)-(k*p.blockK : Nat) := by
  have ks := (envelope_fields M N K p env).2.2.1
  have bk := (positive_tiles p legal).2.2.1
  have small := (ceil_small K p.blockK bk ks).1
  have padded_bound := Nat.div_mul_le_self (K+p.blockK-1) p.blockK
  have cursor := Nat.mul_le_mul_right p.blockK hk
  change k*p.blockK ≤ padded K p.blockK at cursor
  change padded K p.blockK ≤ K+p.blockK-1 at padded_bound
  have tileSmall := (tiles_small p legal).2.2
  unfold remaining
  rw [←BitVec.ofNat_mul]
  change (BitVec.ofInt 32 (K : Int)-BitVec.ofInt 32 (k*p.blockK : Nat)).toInt = _
  rw [BitVec.sub_eq_add_neg,←BitVec.ofInt_neg,←BitVec.ofInt_add]
  exact BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

theorem descending_test (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (k : Nat) (hk : k ≤ ceil K p.blockK) :
    (word 0).slt (remaining K p k) = Bool.true ↔ k*p.blockK < K := by
  rw [BitVec.slt_iff_toInt_lt,toInt (by norm_num),remaining_exact M N K p legal env k hk]
  omega

theorem inner_mask (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (k e : Nat) (hk : k ≤ ceil K p.blockK)
    (he : e < p.blockK) :
    (word e).slt (remaining K p k) = Bool.true ↔ k*p.blockK+e < K := by
  have es := lt_trans he (tiles_small p legal).2.2
  rw [BitVec.slt_iff_toInt_lt,toInt es,remaining_exact M N K p legal env k hk]
  omega

def selectedRow (M N : Nat) (p : Config) (wrapped : Bool) (pid i : Nat) :=
  if wrapped then (row M N p pid i).srem (word M) else row M N p pid i
def selectedColumn (M N : Nat) (p : Config) (wrapped : Bool) (pid j : Nat) :=
  if wrapped then (column M N p pid j).srem (word N) else column M N p pid j
def selectedNaturalRow (M N : Nat) (p : Config) (wrapped : Bool) (pid i : Nat) :=
  if wrapped then naturalRow M N p pid i % M else naturalRow M N p pid i
def selectedNaturalColumn (M N : Nat) (p : Config) (wrapped : Bool) (pid j : Nat) :=
  if wrapped then naturalColumn M N p pid j % N else naturalColumn M N p pid j

theorem selected_coordinates (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (wrapped : Bool) (pid i j : Nat) (hp : pid < count M N p)
    (hi : i < p.blockM) (hj : j < p.blockN) :
    selectedRow M N p wrapped pid i = word (selectedNaturalRow M N p wrapped pid i) ∧
    selectedColumn M N p wrapped pid j = word (selectedNaturalColumn M N p wrapped pid j) ∧
    selectedNaturalRow M N p wrapped pid i < padded M p.blockM ∧
    selectedNaturalColumn M N p wrapped pid j < padded N p.blockN := by
  obtain ⟨rw,cw,_,_⟩ := coordinates M N K p legal env pid i j hp hi hj
  obtain ⟨rb,cb⟩ := coordinate_bounds M N K p legal env pid i j hp hi hj
  have ms := ceil_small M p.blockM (positive_tiles p legal).1 (envelope_fields M N K p env).1
  have ns := ceil_small N p.blockN (positive_tiles p legal).2.1 (envelope_fields M N K p env).2.1
  cases wrapped
  · exact ⟨rw,cw,rb,cb⟩
  · simp only [selectedRow,selectedColumn,selectedNaturalRow,selectedNaturalColumn,↓reduceIte,rw,cw]
    exact ⟨srem _ _ (lt_trans rb ms.2.2.2) ms.1,
      srem _ _ (lt_trans cb ns.2.2.2) ns.1,
      lt_of_le_of_lt (Nat.mod_le _ _) rb,lt_of_le_of_lt (Nat.mod_le _ _) cb⟩

def aOffset (M N K : Nat) (p : Config) (wrapped : Bool) (pid i e k : Nat) : BitVec 32 :=
  selectedRow M N p wrapped pid i*word K+word e+word k*word p.blockK
def bOffset (M N : Nat) (p : Config) (wrapped : Bool) (pid j e k : Nat) : BitVec 32 :=
  word e*word N+selectedColumn M N p wrapped pid j+word k*(word p.blockK*word N)
def cOffset (M N : Nat) (p : Config) (pid i j : Nat) : BitVec 32 :=
  row M N p pid i*word N+column M N p pid j

theorem offset_words (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (wrapped : Bool) (pid i j e k : Nat) (hp : pid < count M N p)
    (hi : i < p.blockM) (hj : j < p.blockN) :
    aOffset M N K p wrapped pid i e k = word (selectedNaturalRow M N p wrapped pid i*K+(k*p.blockK+e)) ∧
    bOffset M N p wrapped pid j e k = word ((k*p.blockK+e)*N+selectedNaturalColumn M N p wrapped pid j) ∧
    cOffset M N p pid i j = word (naturalRow M N p pid i*N+naturalColumn M N p pid j) := by
  obtain ⟨ar,bc,_,_⟩ := selected_coordinates M N K p legal env wrapped pid i j hp hi hj
  obtain ⟨cr,cc,_,_⟩ := coordinates M N K p legal env pid i j hp hi hj
  simp only [aOffset,bOffset,cOffset,ar,bc,cr,cc,word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  constructor
  · congr 1; omega
  constructor
  · congr 1; ring
  · trivial

theorem offset_bounds (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (wrapped : Bool) (pid i j e k : Nat) (hp : pid < count M N p)
    (hi : i < p.blockM) (hj : j < p.blockN) (he : e < p.blockK) (hk : k ≤ ceil K p.blockK) :
    selectedNaturalRow M N p wrapped pid i*K+(k*p.blockK+e) < 2^31 ∧
    (k*p.blockK+e)*N+selectedNaturalColumn M N p wrapped pid j < 2^31 ∧
    naturalRow M N p pid i*N+naturalColumn M N p pid j < 2^31 := by
  obtain ⟨_,_,ar,bc⟩ := selected_coordinates M N K p legal env wrapped pid i j hp hi hj
  obtain ⟨cr,cc⟩ := coordinate_bounds M N K p legal env pid i j hp hi hj
  obtain ⟨_,_,_,_,_,ea,eb,ec⟩ := envelope_fields M N K p env
  have cursor := Nat.mul_le_mul_right p.blockK hk
  change k*p.blockK ≤ padded K p.blockK at cursor
  have a := Nat.mul_le_mul_right K (Nat.le_of_lt ar)
  have b := Nat.mul_le_mul_right N (show k*p.blockK+e ≤ padded K p.blockK+p.blockK by omega)
  have c := Nat.mul_le_mul_right N (Nat.le_of_lt cr)
  exact ⟨by omega,by omega,by omega⟩

theorem offsets_exact (M N K : Nat) (p : Config) (legal : ConfigContract p)
    (env : IndexEnvelope M N K p) (wrapped : Bool) (pid i j e k : Nat) (hp : pid < count M N p)
    (hi : i < p.blockM) (hj : j < p.blockN) (he : e < p.blockK) (hk : k ≤ ceil K p.blockK) :
    (aOffset M N K p wrapped pid i e k).toInt = (selectedNaturalRow M N p wrapped pid i*K+(k*p.blockK+e) : Nat) ∧
    (bOffset M N p wrapped pid j e k).toInt = ((k*p.blockK+e)*N+selectedNaturalColumn M N p wrapped pid j : Nat) ∧
    (cOffset M N p pid i j).toInt = (naturalRow M N p pid i*N+naturalColumn M N p pid j : Nat) := by
  obtain ⟨a,b,c⟩ := offset_words M N K p legal env wrapped pid i j e k hp hi hj
  obtain ⟨ha,hb,hc⟩ := offset_bounds M N K p legal env wrapped pid i j e k hp hi hj he hk
  rw [a,b,c]
  exact ⟨toInt ha,toInt hb,toInt hc⟩

theorem pointer_steps (M N K : Nat) (p : Config) (wrapped : Bool) (pid i j e k : Nat) :
    aOffset M N K p wrapped pid i e (k+1) = aOffset M N K p wrapped pid i e k+word p.blockK ∧
    bOffset M N p wrapped pid j e (k+1) = bOffset M N p wrapped pid j e k+word p.blockK*word N := by
  simp only [aOffset,bOffset,word,BitVec.ofNat_add,BitVec.ofNat_one,add_mul,one_mul]
  constructor <;> simp [add_assoc,add_left_comm,add_comm,mul_assoc,mul_left_comm,mul_comm]

/-- Physical pointer extension/scaling uses the exact checked source offset.
The byte-window premise comes from the Python data_ptr/span checks. -/
theorem accessed_bytes (base count bytes offset : Nat) (hb : 0 < bytes)
    (hi : offset < count) (small : offset < 2^31) (span : base+bytes*count ≤ 2^64) :
    (BitVec.ofNat 64 base+(word offset).signExtend 64*BitVec.ofNat 64 bytes).toNat = base+offset*bytes ∧
    base+offset*bytes+(bytes-1) < base+count*bytes := by
  have extended : (word offset).signExtend 64 = BitVec.ofNat 64 offset := by
    change BitVec.ofInt 64 (word offset).toInt = _
    rw [toInt small]
    rfl
  have bound := Nat.mul_lt_mul_of_pos_right hi hb
  have next := Nat.mul_le_mul_right bytes (show offset+1 ≤ count by omega)
  rw [Nat.add_mul,Nat.one_mul] at next
  rw [extended,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  constructor
  · exact Nat.mod_eq_of_lt (by nlinarith)
  · omega

#axiomsClean blocks_exact
#axiomsClean coordinates
#axiomsClean remaining_exact
#axiomsClean offsets_exact
#axiomsClean accessed_bytes
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FiniteIndex
