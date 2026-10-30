import bench.tritonbench_g.triton_matmul.Proofs.Host
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Index
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 8000000

abbrev word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

/-- Triton scalar ABI for nonnegative tensor metadata: constexpr-one, signed32,
or signed64; the source casts every such operand before arithmetic. -/
def scalarABI (n : Nat) : BitVec 64 :=
  if n=1 then word 1 else if n<2^31 then (BitVec.ofNat 32 n).signExtend 64 else word n
def laneABI (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64
def pidABI := laneABI

theorem word_nat {n : Nat} (h : n<2^64) : (word n).toNat=n := by
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt h]
theorem word_int {n : Nat} (h : n<2^63) : (word n).toInt=(n:Int) := by
  have hn := word_nat (n:=n) (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]
theorem word_msb {n : Nat} (h : n<2^63) : (word n).msb=Bool.false := by
  rw [BitVec.msb_eq_false_iff_two_mul_lt,word_nat (by omega)]
  omega
theorem lane_word {n : Nat} (h : n<2^31) : laneABI n=word n := by
  have hn : (BitVec.ofNat 32 n).toNat=n := by
    simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (show n<2^32 by omega)]
  have hi : (BitVec.ofNat 32 n).toInt=(n:Int) := by
    rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]
  change BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt=word n
  rw [hi,BitVec.ofInt_natCast]
theorem scalar_word (n : Nat) : scalarABI n=word n := by
  unfold scalarABI
  split_ifs with h hs
  · subst n; rfl
  · exact lane_word hs
  · rfl
theorem signed_div {a b : Nat} (ha : a<2^63) (hb : b<2^63) :
    (word a).sdiv (word b)=word (a/b) := by
  rw [BitVec.sdiv_eq,word_msb ha,word_msb hb]
  apply BitVec.eq_of_toNat_eq
  change (word a / word b).toNat=(word (a/b)).toNat
  simp only [BitVec.toNat_udiv,word_nat (by omega : a<2^64),word_nat (by omega : b<2^64),BitVec.toNat_ofNat]
  exact (Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.div_le_self a b) (show a<2^64 by omega))).symm
theorem signed_rem {a b : Nat} (ha : a<2^63) (hb : b<2^63) :
    (word a).srem (word b)=word (a%b) := by
  rw [BitVec.srem_eq,word_msb ha,word_msb hb]
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_umod,word_nat (by omega : a<2^64),word_nat (by omega : b<2^64),BitVec.toNat_ofNat]
  exact (Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.mod_le a b) (show a<2^64 by omega))).symm
theorem signed_lt {a b : Nat} (ha : a<2^63) (hb : b<2^63) :
    (word a).slt (word b)=decide (a<b) := by
  simp only [BitVec.slt_eq_decide,word_int ha,word_int hb]
  simp
theorem subtract {a b : Nat} (ha : a<2^63) (h : b≤a) :
    word a-word b=word (a-b) := BitVec.ofNat_sub_ofNat_of_le a b (by omega) h
def smin (a b : BitVec 64) := if a.slt b then a else b
theorem signed_min {a b : Nat} (ha : a<2^63) (hb : b<2^63) :
    smin (word a) (word b)=word (min a b) := by
  rw [smin,signed_lt ha hb]
  by_cases h : a<b
  · simp [h,Nat.min_eq_left (by omega : a≤b)]
  · simp [h,Nat.min_eq_right (by omega : b≤a)]

def ceiling (value : BitVec 64) (block : Nat) : BitVec 64 :=
  value.sdiv (word block) + if value.srem (word block)==word 0 then word 0 else word 1
theorem ceiling_word {n block : Nat} (hn : n<2^63) (hb : block<2^63) :
    ceiling (scalarABI n) block=word (safeCeil n block) := by
  rw [ceiling,scalar_word,signed_div hn hb,signed_rem hn hb]
  have hm : n%block<2^64 := lt_of_le_of_lt (Nat.mod_le n block) (by omega)
  by_cases h : n%block=0
  · simp [h,safeCeil,BitVec.ofNat_add_ofNat]
  · have hz : word (n%block)≠word 0 := by
      intro he
      have := congrArg BitVec.toNat he
      simp only [word_nat hm,BitVec.toNat_ofNat,Nat.zero_mod] at this
      exact h this
    simp [h,hz,safeCeil,BitVec.ofNat_add_ofNat]


structure Control where
  (pid numM numN inGroup group first size row col startM startN blocks : BitVec 64)
  deriving DecidableEq
def control (r : Request) (pid : Nat) : Control := Id.run do
  let p := pidABI pid
  let nm := ceiling (scalarABI r.rows) 128
  let nn := ceiling (scalarABI r.columns) 256
  let ng := word 8*nn
  let group := p.sdiv ng
  let first := group*word 8
  let size := smin (nm-first) (word 8)
  let row := first+p.srem size
  let col := (p.srem ng).sdiv size
  return ⟨p,nm,nn,ng,group,first,size,row,col,row*word 128,col*word 256,
    ceiling (scalarABI r.inner) (SourceScope.blockK r.storage)⟩
def naturalControl (r : Request) (pid : Nat) : Control :=
  ⟨word pid,word (tileRows r),word (tileColumns r),word (8*tileColumns r),
    word (pid/(8*tileColumns r)),word (Repair.groupBase (tileRows r) (tileColumns r) 8 pid),
    word (Repair.groupSize (tileRows r) (tileColumns r) 8 pid),
    word (tileRow (tileRows r) (tileColumns r) 8 pid),
    word (tileCol (tileRows r) (tileColumns r) 8 pid),
    word (tileRow (tileRows r) (tileColumns r) 8 pid*128),
    word (tileCol (tileRows r) (tileColumns r) 8 pid*256),word r.parameters.numKBlocks⟩

theorem control_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (hlaunch : programs r<2^31) :
    control r pid=naturalControl r pid := by
  obtain ⟨hgrid,hng,hngw,hs,hfirst,hr,hc⟩ := control_machine r s t pid hp
  have hm : tileRows r<2^63 := by have := t.dimM; unfold tileRows; omega
  have hn : tileColumns r<2^63 := by have := t.dimN; unfold tileColumns; omega
  have hg : pid/(8*tileColumns r)<2^63 := lt_of_le_of_lt (Nat.div_le_self _ _) (lt_trans hp hgrid)
  have hf : Repair.groupBase (tileRows r) (tileColumns r) 8 pid<2^63 := lt_trans hfirst hm
  have hsize : Repair.groupSize (tileRows r) (tileColumns r) 8 pid<2^63 := by
    have : Repair.groupSize (tileRows r) (tileColumns r) 8 pid≤8 := min_le_right _ _
    omega
  have rem : pid%(8*tileColumns r)<2^63 := lt_trans (Nat.mod_lt _ hng) hngw
  have cm : ceiling (scalarABI r.rows) 128=word (tileRows r) := by
    rw [ceiling_word t.dimM (by decide),(ceiling_normalization r.rows).2.1]; rfl
  have cn : ceiling (scalarABI r.columns) 256=word (tileColumns r) := by
    rw [ceiling_word t.dimN (by decide),(ceiling_normalization r.columns).2.2]; rfl
  have ck : ceiling (scalarABI r.inner) (SourceScope.blockK r.storage)=word r.parameters.numKBlocks := by
    cases h : r.storage <;> simp only [h,SourceScope.blockK]
    · rw [ceiling_word t.dimK (by decide),(ceiling_normalization r.inner).1]; simp [Request.parameters,SourceScope.parameters,SourceScope.blockK,h,Nat.add_sub_assoc]
    · rw [ceiling_word t.dimK (by decide),(ceiling_normalization r.inner).2.1]; simp [Request.parameters,SourceScope.parameters,SourceScope.blockK,h,Nat.add_sub_assoc]
  have first : (word (pid/(8*tileColumns r))*word 8)=
      word (Repair.groupBase (tileRows r) (tileColumns r) 8 pid) := by
    rw [BitVec.ofNat_mul_ofNat]; rfl
  have size : smin (word (tileRows r)-word (Repair.groupBase (tileRows r) (tileColumns r) 8 pid)) (word 8)=
      word (Repair.groupSize (tileRows r) (tileColumns r) 8 pid) := by
    rw [subtract hm (Nat.le_of_lt hfirst),signed_min (by omega) (by decide)]
    rfl
  dsimp only [control,Id.run]
  rw [show pidABI pid=word pid from lane_word (lt_trans hp hlaunch),cm,cn,ck,
    BitVec.ofNat_mul_ofNat,signed_div (lt_trans hp hgrid) hngw,first,size,
    signed_rem (lt_trans hp hgrid) hsize,signed_rem (lt_trans hp hgrid) hngw,
    signed_div rem hsize,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat,BitVec.ofNat_mul_ofNat]
  rfl

/-- Dynamic scf.for induction variable and backedge execute in signed64. -/
def counter : Nat → BitVec 64
  | 0 => word 0
  | n+1 => counter n+word 1
theorem counter_word (n : Nat) : counter n=word n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [counter,ih,BitVec.ofNat_add_ofNat]

def mask (limit : Nat) (base : BitVec 64) (lane : Nat) :=
  (laneABI lane).slt (scalarABI limit-base)
def clamped (limit : Nat) (base : BitVec 64) (lane : Nat) : BitVec 64 :=
  (if mask limit base lane then base else word 0)+
  (if mask limit base lane then laneABI lane else word 0)

theorem mask_word {limit base lane : Nat} (hl : limit<2^63)
    (hb : base≤limit) (hn : lane<2^31) :
    mask limit (word base) lane=decide (base+lane<limit) := by
  rw [mask,lane_word hn,scalar_word,subtract hl hb,signed_lt (by omega) (by omega)]
  simp only [decide_eq_decide]
  omega
theorem clamped_word {limit base lane : Nat} (hl : limit<2^63)
    (hb : base≤limit) (hn : lane<2^31) :
    clamped limit (word base) lane=word (safeCoordinate limit (base+lane)) := by
  rw [clamped,mask_word hl hb hn,lane_word hn]
  by_cases h : base+lane<limit <;> simp [h,safeCoordinate,BitVec.ofNat_add_ofNat]

theorem blocks_bound {α : Type} (r : Request) (s : State α) (t : TensorAPI r s) :
    r.parameters.numKBlocks<2^63 := by
  have h := ceiling_machine r.inner t.dimK
  have n := ceiling_normalization r.inner
  cases hs : r.storage <;> simp [Request.parameters,SourceScope.parameters,SourceScope.blockK,hs] <;> omega

theorem loop_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (block : Nat) (hb : block≤r.parameters.numKBlocks) :
    (counter block).toInt=(block:Int) ∧
    (counter block).slt (ceiling (scalarABI r.inner) (SourceScope.blockK r.storage))=
      decide (block<r.parameters.numKBlocks) ∧
    (block<r.parameters.numKBlocks → (counter block+word 1).toInt=((block+1:Nat):Int)) := by
  have h := blocks_bound r s t
  have hk : ceiling (scalarABI r.inner) (SourceScope.blockK r.storage)=word r.parameters.numKBlocks := by
    cases hs : r.storage <;> simp only [hs,SourceScope.blockK]
    · rw [ceiling_word t.dimK (by decide),(ceiling_normalization r.inner).1]; simp [Request.parameters,SourceScope.parameters,SourceScope.blockK,hs,Nat.add_sub_assoc]
    · rw [ceiling_word t.dimK (by decide),(ceiling_normalization r.inner).2.1]; simp [Request.parameters,SourceScope.parameters,SourceScope.blockK,hs,Nat.add_sub_assoc]
  rw [counter_word,hk]
  refine ⟨word_int (by omega),signed_lt (by omega) h,?_⟩
  intro hb'
  rw [BitVec.ofNat_add_ofNat]
  exact word_int (by omega)

def rowCoordinate (r : Request) (pid lane : Nat) := tileRow (tileRows r) (tileColumns r) 8 pid*128+lane
def colCoordinate (r : Request) (pid lane : Nat) := tileCol (tileRows r) (tileColumns r) 8 pid*256+lane
def innerCoordinate (r : Request) (block lane : Nat) := block*SourceScope.blockK r.storage+lane
def rowBits (r : Request) (pid lane : Nat) := clamped r.rows (control r pid).startM lane
def colBits (r : Request) (pid lane : Nat) := clamped r.columns (control r pid).startN lane
def innerBits (r : Request) (block lane : Nat) := clamped r.inner (counter block*word (SourceScope.blockK r.storage)) lane
def innerMask (r : Request) (block lane : Nat) := mask r.inner (counter block*word (SourceScope.blockK r.storage)) lane
def storeMask (r : Request) (pid i j : Nat) := mask r.rows (control r pid).startM i && mask r.columns (control r pid).startN j

theorem row_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid i : Nat) (hp : pid<programs r) (hl : programs r<2^31) (hi : i<128) :
    rowBits r pid i=word (safeCoordinate r.rows (rowCoordinate r pid i)) := by
  rw [rowBits,control_word r s t pid hp hl]
  exact clamped_word t.dimM (Nat.le_of_lt (control_machine r s t pid hp).2.2.2.2.2.1) (by omega)
theorem col_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid j : Nat) (hp : pid<programs r) (hl : programs r<2^31) (hj : j<256) :
    colBits r pid j=word (safeCoordinate r.columns (colCoordinate r pid j)) := by
  rw [colBits,control_word r s t pid hp hl]
  exact clamped_word t.dimN (Nat.le_of_lt (control_machine r s t pid hp).2.2.2.2.2.2) (by omega)
theorem inner_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (block k : Nat) (hb : block<r.parameters.numKBlocks) (hk : k<SourceScope.blockK r.storage) :
    innerBits r block k=word (safeCoordinate r.inner (innerCoordinate r block k)) := by
  have bound : SourceScope.blockK r.storage≤128 := by cases r.storage <;> decide
  rw [innerBits,counter_word,BitVec.ofNat_mul_ofNat]
  exact clamped_word t.dimK (Nat.le_of_lt (inner_base_machine r s t block hb).1) (by omega)

theorem masks_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid i j : Nat) (hp : pid<programs r) (hl : programs r<2^31) (hi : i<128) (hj : j<256) :
    storeMask r pid i j=decide (rowCoordinate r pid i<r.rows ∧ colCoordinate r pid j<r.columns) := by
  rw [storeMask,control_word r s t pid hp hl]
  dsimp only [naturalControl]
  rw [mask_word t.dimM (Nat.le_of_lt (control_machine r s t pid hp).2.2.2.2.2.1) (by omega),
    mask_word t.dimN (Nat.le_of_lt (control_machine r s t pid hp).2.2.2.2.2.2) (by omega)]
  simp only [rowCoordinate,colCoordinate,Bool.decide_and]
theorem inner_mask_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (block k : Nat) (hb : block<r.parameters.numKBlocks) (hk : k<SourceScope.blockK r.storage) :
    innerMask r block k=decide (innerCoordinate r block k<r.inner) := by
  have bound : SourceScope.blockK r.storage≤128 := by cases r.storage <;> decide
  rw [innerMask,counter_word,BitVec.ofNat_mul_ofNat]
  exact mask_word t.dimK (Nat.le_of_lt (inner_base_machine r s t block hb).1) (by omega)

def aOffset (r : Request) (pid block i k : Nat) :=
  rowBits r pid i*scalarABI r.strideAM+innerBits r block k*scalarABI r.strideAK
def bOffset (r : Request) (pid block k j : Nat) :=
  innerBits r block k*scalarABI r.strideBK+colBits r pid j*scalarABI r.strideBN
def addptr (base : BitVec 64) (bytes : Nat) (offset : BitVec 64) : BitVec 64 :=
  base+BitVec.ofInt 64 ((bytes:Int)*offset.toInt)
def aPointer (r : Request) (base pid block i k : Nat) := addptr (word base) (elementBytes r.storage) (aOffset r pid block i k)
def bPointer (r : Request) (base pid block k j : Nat) := addptr (word base) (elementBytes r.storage) (bOffset r pid block k j)
def cPointer (r : Request) (base pid i j : Nat) :=
  addptr (addptr (word base) (elementBytes r.storage) (scalarABI (max r.columns 1)*rowBits r pid i))
    (elementBytes r.storage) (scalarABI 1*colBits r pid j)

theorem signed_addptr (base : BitVec 64) (bytes : Nat) (offset : BitVec 64) :
    addptr base bytes offset=base+word bytes*offset := by
  rw [addptr,BitVec.ofInt_mul,BitVec.ofInt_toInt,BitVec.ofInt_natCast]
/-- The Python load pointer uses one addptr with a parenthesized sum; the typed
AST uses two successive pointer additions. Their finite byte residues agree
for every pair of offsets, before any no-wrap bounds are used. -/
theorem pointer_add_associativity (base x y : BitVec 64) (bytes : Nat) :
    addptr (addptr base bytes x) bytes y=addptr base bytes (x+y) := by
  simp only [signed_addptr,BitVec.mul_add,BitVec.add_assoc]

theorem offsets_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid block i j k : Nat) (hp : pid<programs r) (hl : programs r<2^31)
    (hb : block<r.parameters.numKBlocks) (hi : i<128) (hj : j<256) (hk : k<SourceScope.blockK r.storage) :
    aOffset r pid block i k=word (safeAOffset r (rowCoordinate r pid i) (innerCoordinate r block k)) ∧
    bOffset r pid block k j=word (safeBOffset r (innerCoordinate r block k) (colCoordinate r pid j)) := by
  simp only [aOffset,bOffset,row_word r s t pid i hp hl hi,col_word r s t pid j hp hl hj,
    inner_word r s t block k hb hk,scalar_word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  exact ⟨rfl,rfl⟩

theorem pointers_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid block i j k : Nat) (hp : pid<programs r) (hl : programs r<2^31)
    (hb : block<r.parameters.numKBlocks) (hi : i<128) (hj : j<256) (hk : k<SourceScope.blockK r.storage) :
    (aPointer r t.baseA pid block i k).toNat=t.baseA+elementBytes r.storage*safeAOffset r (rowCoordinate r pid i) (innerCoordinate r block k) ∧
    (bPointer r t.baseB pid block k j).toNat=t.baseB+elementBytes r.storage*safeBOffset r (innerCoordinate r block k) (colCoordinate r pid j) ∧
    (aOffset r pid block i k).toInt=(safeAOffset r (rowCoordinate r pid i) (innerCoordinate r block k):Int) ∧
    (bOffset r pid block k j).toInt=(safeBOffset r (innerCoordinate r block k) (colCoordinate r pid j):Int) := by
  obtain ⟨hr,hc⟩ := nonempty_program r pid hp
  have hinner : 0<r.inner := by have := (inner_base_machine r s t block hb).1; omega
  obtain ⟨ha,hb'⟩ := offsets_word r s t pid block i j k hp hl hb hi hj hk
  have sa := safeA_machine r s t (rowCoordinate r pid i) (innerCoordinate r block k) hr hinner
  have sb := safeB_machine r s t (innerCoordinate r block k) (colCoordinate r pid j) hinner hc
  rw [aPointer,bPointer,signed_addptr,signed_addptr,ha,hb']
  simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  exact ⟨word_nat sa.2.2.2.2,word_nat sb.2.2.2.2,word_int sa.2.2.1,word_int sb.2.2.1⟩

theorem output_pointer_word {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid i j : Nat) (hp : pid<programs r) (hl : programs r<2^31) (hi : i<128) (hj : j<256) :
    (cPointer r t.baseC pid i j).toNat=t.baseC+elementBytes r.storage*safeCOffset r (rowCoordinate r pid i) (colCoordinate r pid j) := by
  obtain ⟨hr,hc⟩ := nonempty_program r pid hp
  have h := (safeC_machine r s t (rowCoordinate r pid i) (colCoordinate r pid j) hr hc).2.2.2.2
  simp only [cPointer,signed_addptr,row_word r s t pid i hp hl hi,col_word r s t pid j hp hl hj,
    scalar_word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  have he : t.baseC+elementBytes r.storage*(max r.columns 1*safeCoordinate r.rows (rowCoordinate r pid i))+
      elementBytes r.storage*(1*safeCoordinate r.columns (colCoordinate r pid j))=
      t.baseC+elementBytes r.storage*safeCOffset r (rowCoordinate r pid i) (colCoordinate r pid j) := by
    unfold safeCOffset; ring
  rw [he]
  exact word_nat h

theorem source_coordinates {α : Type} (r : Request) (s : State α) (pid : Nat)
    (i : Fin (BM r.parameters)) (j : Fin (BN r.parameters)) :
    FPValue.row r.parameters (instanceState r pid s) i=rowCoordinate r pid i.val ∧
    FPValue.col r.parameters (instanceState r pid s) j=colCoordinate r pid j.val := by
  simp only [FPValue.row,FPValue.col,source_pidM,source_pidN,rowCoordinate,colCoordinate]
  exact ⟨rfl,rfl⟩

/-- Direct binding to the changed typed A pointer expression, not merely a
separate address bound. The arbitrary inactive lanes remain clamped. -/
theorem typed_a_offset {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid block : Nat) (i : AIndex r.parameters) (hp : pid<programs r)
    (hl : programs r<2^31) (hb : block<r.parameters.numKBlocks) :
    aOffset r pid block i.1.val i.2.1.val=word (
      aRow r.parameters (instanceState r pid s) i.1*r.strideAM+
      ((if i.2.1.val<r.inner-block*SourceScope.blockK r.storage then block*SourceScope.blockK r.storage else 0)+
       (if i.2.1.val<r.inner-block*SourceScope.blockK r.storage then i.2.1.val else 0))*r.strideAK) := by
  rw [candidate_a_pointer]
  have h := (offsets_word r s t pid block i.1.val 0 i.2.1.val hp hl hb i.1.isLt (by decide) i.2.1.isLt).1
  simpa only [FPValue.row,source_pidM,rowCoordinate,innerCoordinate] using h

theorem typed_b_offset {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid block : Nat) (i : BIndex r.parameters) (hp : pid<programs r)
    (hl : programs r<2^31) (hb : block<r.parameters.numKBlocks) :
    bOffset r pid block i.1.val i.2.1.val=word (
      ((if i.1.val<r.inner-block*SourceScope.blockK r.storage then block*SourceScope.blockK r.storage else 0)+
       (if i.1.val<r.inner-block*SourceScope.blockK r.storage then i.1.val else 0))*r.strideBK+
      bCol r.parameters (instanceState r pid s) i.2.1*r.strideBN) := by
  rw [candidate_b_pointer]
  have h := (offsets_word r s t pid block 0 i.2.1.val i.1.val hp hl hb (by decide) i.2.1.isLt i.1.isLt).2
  simpa only [FPValue.col,source_pidN,colCoordinate,innerCoordinate] using h

theorem typed_output {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (i : CIndex r.parameters) (hp : pid<programs r) (hl : programs r<2^31) :
    (cPointer r t.baseC pid i.1.val i.2.1.val).toNat=
      t.baseC+elementBytes r.storage*Source.cSafeOffset r.parameters (instanceState r pid s) i ∧
    storeMask r pid i.1.val i.2.1.val=decide (active r.parameters (instanceState r pid s) i) := by
  rw [candidate_store_offset]
  obtain ⟨hr,hc⟩ := source_coordinates r s pid i.1 i.2.1
  rw [hr,hc]
  refine ⟨output_pointer_word r s t pid i.1.val i.2.1.val hp hl i.1.isLt i.2.1.isLt,?_⟩
  rw [masks_word r s t pid i.1.val i.2.1.val hp hl i.1.isLt i.2.1.isLt]
  simp only [active,hr,hc]
  rfl

theorem metadata_exact {α : Type} (r : Request) (s : State α) (t : TensorAPI r s) :
    ∀ n ∈ [r.rows,r.columns,r.inner,r.strideAM,r.strideAK,r.strideBK,r.strideBN,max r.columns 1,1],
      (scalarABI n).toInt=(n:Int) := by
  intro n hn
  rw [scalar_word]
  apply word_int
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hn
  have hm := t.dimM; have hn' := t.dimN; have hk := t.dimK
  have ha := t.strideAM; have hb := t.strideAK; have hc := t.strideBK; have hd := t.strideBN
  rcases hn with h | h | h | h | h | h | h | h | h <;> subst n <;> omega

theorem whole_word_bytes (base bytes extent offset : Nat)
    (hspan : base+bytes*extent≤2^64) (hoff : offset<extent) :
    base+bytes*offset+bytes≤2^64 := by
  have hm := Nat.mul_le_mul_left bytes (Nat.succ_le_of_lt hoff)
  nlinarith

/-- The bridge executes the actual scalar ABI, signed arithmetic/comparisons,
clamps, K induction/backedge and byte pointer formation. It accompanies the
typed complete execution; it is not a new compiler or numerical hypothesis. -/
structure Correspondence {α : Type} (r : Request) (s : State α) (t : TensorAPI r s) : Prop where
  metadata : ∀ n ∈ [r.rows,r.columns,r.inner,r.strideAM,r.strideAK,r.strideBK,r.strideBN,max r.columns 1,1],
    (scalarABI n).toInt=(n:Int)
  control : ∀ pid, pid<programs r → Index.control r pid=naturalControl r pid
  loop : ∀ block, block≤r.parameters.numKBlocks →
    (counter block).toInt=(block:Int) ∧
    (counter block).slt (ceiling (scalarABI r.inner) (SourceScope.blockK r.storage))=decide (block<r.parameters.numKBlocks) ∧
    (block<r.parameters.numKBlocks → (counter block+word 1).toInt=((block+1:Nat):Int))
  coordinates : ∀ pid (i : Fin (BM r.parameters)) (j : Fin (BN r.parameters)),
    FPValue.row r.parameters (instanceState r pid s) i=rowCoordinate r pid i.val ∧
    FPValue.col r.parameters (instanceState r pid s) j=colCoordinate r pid j.val
  inputs : ∀ pid block i j k, pid<programs r → block<r.parameters.numKBlocks →
    i<128 → j<256 → k<SourceScope.blockK r.storage →
    (aPointer r t.baseA pid block i k).toNat=t.baseA+elementBytes r.storage*safeAOffset r (rowCoordinate r pid i) (innerCoordinate r block k) ∧
    (bPointer r t.baseB pid block k j).toNat=t.baseB+elementBytes r.storage*safeBOffset r (innerCoordinate r block k) (colCoordinate r pid j) ∧
    (aOffset r pid block i k).toInt=(safeAOffset r (rowCoordinate r pid i) (innerCoordinate r block k):Int) ∧
    (bOffset r pid block k j).toInt=(safeBOffset r (innerCoordinate r block k) (colCoordinate r pid j):Int)
  inputMask : ∀ block k, block<r.parameters.numKBlocks → k<SourceScope.blockK r.storage →
    innerMask r block k=decide (innerCoordinate r block k<r.inner)
  output : ∀ pid i j, pid<programs r → i<128 → j<256 →
    (cPointer r t.baseC pid i j).toNat=t.baseC+elementBytes r.storage*safeCOffset r (rowCoordinate r pid i) (colCoordinate r pid j)
  outputMask : ∀ pid i j, pid<programs r → i<128 → j<256 →
    storeMask r pid i j=decide (rowCoordinate r pid i<r.rows ∧ colCoordinate r pid j<r.columns)
  fullABytes : ∀ row k, 0<r.rows → 0<r.inner →
    t.baseA+elementBytes r.storage*safeAOffset r row k+elementBytes r.storage≤2^64
  fullBBytes : ∀ k col, 0<r.inner → 0<r.columns →
    t.baseB+elementBytes r.storage*safeBOffset r k col+elementBytes r.storage≤2^64
  fullCBytes : ∀ row col, 0<r.rows → 0<r.columns →
    t.baseC+elementBytes r.storage*safeCOffset r row col+elementBytes r.storage≤2^64

theorem correspondence {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (hl : programs r<2^31) : Correspondence r s t where
  metadata := metadata_exact r s t
  control := fun pid hp => control_word r s t pid hp hl
  loop := loop_word r s t
  coordinates := source_coordinates r s
  inputs := fun pid block i j k hp hb hi hj hk => pointers_word r s t pid block i j k hp hl hb hi hj hk
  inputMask := inner_mask_word r s t
  output := fun pid i j hp hi hj => output_pointer_word r s t pid i j hp hl hi hj
  outputMask := fun pid i j hp hi hj => masks_word r s t pid i j hp hl hi hj
  fullABytes := fun row k hr hk => whole_word_bytes _ _ _ _ t.spanA (safeA_bound r s t row k hr hk)
  fullBBytes := fun k col hk hc => whole_word_bytes _ _ _ _ t.spanB (safeB_bound r s t k col hk hc)
  fullCBytes := fun row col hr hc => whole_word_bytes _ _ _ _ t.spanC (safeC_bound r row col hr hc)

#axiomsClean correspondence
#axiomsClean control_word
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Index
