import bench.tritonbench_g.iv_dependent_matmul.Proofs.SourceDefinitions
import bench.tritonbench_g.iv_dependent_matmul.Sources.Index64

/-! Actual 64-bit source expressions and pointer-register transitions. The
projection below proves ABI casts, signed division/remainder/comparisons,
ceil-loop control, masked loads/stores and all five staged-pointer schedules.
Inactive pointer arithmetic is modular; only actual accesses require storage.
No floating-point premise is used by any declaration in this module. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.Native
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
noncomputable section
set_option maxHeartbeats 4000000

theorem signed_word (n : Nat) (h : n<2^63) : (word n).toInt=(n : Int) := by
  rw [show word n=BitVec.ofInt 64 (n : Int) from (BitVec.ofInt_natCast _ _).symm]
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · norm_num
  · exact_mod_cast h

theorem unsigned_word (n : Nat) (h : n<2^64) : (word n).toNat=n := by
  exact Nat.mod_eq_of_lt h

theorem nonnegative_word (n : Nat) (h : n<2^63) : (word n).msb=Bool.false := by
  apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
  rw [unsigned_word n (by omega)]
  omega

theorem cast_i32 (n : Nat) (h : n<2^31) : castI32 n=word n := by
  unfold castI32 BitVec.signExtend
  have hs : (BitVec.ofNat 32 n).toInt=(n : Int) := by
    rw [← BitVec.ofInt_natCast]
    apply BitVec.toInt_ofInt_eq_self (by decide)
    · norm_num
    · exact_mod_cast h
  rw [hs]
  exact BitVec.ofInt_natCast _ _

/-- JIT specialization of ordinary nonnegative Torch integer metadata. -/
theorem scalar_word (n : Nat) : scalarABI n=word n := by
  unfold scalarABI
  split
  · rename_i h; rw [h]
  · split
    · rename_i h; exact cast_i32 n h
    · rfl

theorem sdiv_word (x y : Nat) (hx : x<2^63) (hy : y<2^63) :
    (word x).sdiv (word y)=word (x/y) := by
  rw [BitVec.sdiv_eq,nonnegative_word x hx,nonnegative_word y hy]
  apply BitVec.eq_of_toNat_eq
  change (word x / word y).toNat=(word (x/y)).toNat
  simp only [BitVec.toNat_udiv,unsigned_word x (by omega),unsigned_word y (by omega)]
  exact (unsigned_word (x/y) (lt_of_le_of_lt (Nat.div_le_self _ _) (by omega))).symm

theorem srem_word (x y : Nat) (hx : x<2^63) (hy : y<2^63) :
    (word x).srem (word y)=word (x%y) := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.srem_eq,nonnegative_word x hx,nonnegative_word y hy]
  simp only [BitVec.toNat_umod,unsigned_word x (by omega),unsigned_word y (by omega)]
  exact (unsigned_word (x%y) (lt_of_le_of_lt (Nat.mod_le _ _) (by omega))).symm

theorem slt_word (x y : Nat) (hx : x<2^63) (hy : y<2^63) :
    (word x).slt (word y)=decide (x<y) := by
  simp only [BitVec.slt_eq_decide,signed_word x hx,signed_word y hy]
  simp

theorem ceil_word (n b : Nat) (hb : 0<b) (h : n+b<2^63) :
    ceilWord n b=word ((n+b-1)/b) := by
  unfold ceilWord
  rw [scalar_word,BitVec.ofNat_add_ofNat,BitVec.ofNat_sub_ofNat_of_le _ _ (by norm_num) (by omega)]
  exact sdiv_word _ _ (by omega) (by omega)

theorem coordinates (r : Request) (s : State Word) (a : Allocation r s)
    (d : Index64.Addressable r s a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) :
    (∀ i : Fin r.blockRows, rowWord r pid i.val=word (row r.parameters (instanceState pid s) i)) ∧
    (∀ j : Fin r.blockColumns, colWord r pid j.val=word (col r.parameters (instanceState pid s) j)) := by
  have hs := Index64.shape_control r s a d l pid hp
  have hc := (Index64.counts_control r s a d l pid hp).2.1
  change (r.columns+r.blockColumns-1)/r.blockColumns<2^63 at hc
  have hpid : pid<2^31 := lt_trans hp l.grid
  have hp64 : pid<2^63 := by omega
  constructor
  · intro i
    have hi : i.val<2^31 := lt_of_lt_of_le i.isLt l.rowsBlock
    simp only [rowWord,cast_i32 pid hpid,cast_i32 i.val hi,
      ceil_word _ _ r.positiveColumnsBlock hs.2.1]
    rw [sdiv_word _ _ hp64 hc]
    simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
    rfl
  · intro j
    have hj : j.val<2^31 := lt_of_lt_of_le j.isLt l.columnsBlock
    simp only [colWord,cast_i32 pid hpid,cast_i32 j.val hj,
      ceil_word _ _ r.positiveColumnsBlock hs.2.1]
    rw [srem_word _ _ hp64 hc]
    simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
    rfl

theorem load_mask (r : Request) (s : State Word) (a : Allocation r s)
    (d : Index64.Addressable r s a) (l : Index64.Launchable r)
    (pid k : Nat) (hp : pid<programs r) (hk : k<count r.parameters) (lane : Fin r.blockInner) :
    loadMask r k lane.val=decide (lane.val<r.inner-k*r.blockInner) := by
  have hs := Index64.shape_control r s a d l pid hp
  have hi := source_subtraction r k hk
  have hl : lane.val<2^31 := lt_of_lt_of_le lane.isLt l.innerBlock
  unfold loadMask
  rw [cast_i32 lane.val hl,scalar_word,BitVec.ofNat_mul_ofNat]
  rw [BitVec.ofNat_sub_ofNat_of_le _ _ (by omega) (by omega)]
  apply slt_word <;> omega

theorem store_mask (r : Request) (s : State Word) (a : Allocation r s)
    (d : Index64.Addressable r s a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) (q : CIndex r) :
    storeMask r pid q=active r.parameters (instanceState pid s) q := by
  have hc := coordinates r s a d l pid hp
  have hb := Index64.program_coordinates r s l pid hp
  have hs := Index64.shape_control r s a d l pid hp
  unfold storeMask
  rw [hc.1 q.1,hc.2 q.2.1,scalar_word,scalar_word]
  rw [slt_word _ _ (hb.1 _) (by omega),slt_word _ _ (hb.2 _) (by omega)]
  rfl

theorem signed_byte_offset (x : I64) : BitVec.ofInt 64 (4*x.toInt)=word 4*x := by
  rw [BitVec.ofInt_mul,BitVec.ofInt_toInt]
  rfl

theorem a_base (r : Request) (s : State Word) (a : Allocation r s)
    (d : Index64.Addressable r s a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) (i : TileIndex [r.blockRows,r.blockInner]) :
    aBaseWord r pid i=word (aBase r.parameters (instanceState pid s) i) := by
  have hc := coordinates r s a d l pid hp
  have hb := (Index64.program_coordinates r s l pid hp).1 i.1
  have hs := (Index64.shape_control r s a d l pid hp).1
  unfold aBaseWord
  rw [hc.1 i.1,scalar_word,scalar_word,scalar_word,
    cast_i32 i.2.1.val (lt_of_lt_of_le i.2.1.isLt l.innerBlock)]
  rw [srem_word _ _ hb (by omega)]
  simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  rfl

theorem b_base (r : Request) (s : State Word) (a : Allocation r s)
    (d : Index64.Addressable r s a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) (i : TileIndex [r.blockInner,r.blockColumns]) :
    bBaseWord r pid i=word (bBase r.parameters (instanceState pid s) i) := by
  have hc := coordinates r s a d l pid hp
  have hb := (Index64.program_coordinates r s l pid hp).2 i.2.1
  have hs := (Index64.shape_control r s a d l pid hp).2.1
  unfold bBaseWord
  rw [hc.2 i.2.1,scalar_word,scalar_word,scalar_word,
    cast_i32 i.1.val (lt_of_lt_of_le i.1.isLt l.innerBlock)]
  rw [srem_word _ _ hb (by omega)]
  simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  rfl

theorem a_pointer (r : Request) (s : State Word) (a : Allocation r s)
    (d : Index64.Addressable r s a) (l : Index64.Launchable r)
    (pid k : Nat) (hp : pid<programs r) (i : TileIndex [r.blockRows,r.blockInner]) :
    aAt r a.baseA pid k i=word (a.baseA+4*aOffset r.parameters (instanceState pid s) k i) := by
  unfold aAt
  rw [a_base r s a d l pid hp i,scalar_word]
  simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  congr 1
  simp only [aOffset,Request.parameters]
  ring

theorem b_pointer (r : Request) (s : State Word) (a : Allocation r s)
    (d : Index64.Addressable r s a) (l : Index64.Launchable r)
    (pid k : Nat) (hp : pid<programs r) (i : TileIndex [r.blockInner,r.blockColumns]) :
    bAt r a.baseB pid k i=word (a.baseB+4*bOffset r.parameters (instanceState pid s) k i) := by
  unfold bAt
  rw [b_base r s a d l pid hp i,scalar_word]
  simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  congr 1
  simp only [bOffset,Request.parameters]
  ring

theorem c_pointer (r : Request) (s : State Word) (a : Allocation r s)
    (d : Index64.Addressable r s a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) (i : CIndex r) :
    cAt r a.baseC pid i=word (a.baseC+4*outputOffset r.parameters (instanceState pid s) i) := by
  have hc := coordinates r s a d l pid hp
  unfold cAt
  rw [scalar_word,scalar_word,hc.1 i.1,hc.2 i.2.1]
  simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  congr 1
  simp only [outputOffset,Request.parameters]
  ring

def CacheInvariant (r : Request) (ba bb pid k : Nat) (c : Cache r) : Prop :=
  let now := c.ac=aAt r ba pid k ∧ c.bc=bAt r bb pid k
  let next := c.an=aAt r ba pid (k+1) ∧ c.bn=bAt r bb pid (k+1)
  let nextNext := c.ann=aAt r ba pid (k+2) ∧ c.bnn=bAt r bb pid (k+2)
  match r.schedule with
  | .preLoad => True
  | .postLoad => now
  | .mixed => c.bc=bAt r bb pid k
  | .two => now ∧ next
  | .three => now ∧ next ∧ nextNext

theorem cache_invariant (r : Request) (ba bb pid k : Nat) :
    CacheInvariant r ba bb pid k (cacheAt r ba bb pid k) := by
  induction k with
  | zero => cases h : r.schedule <;> simp [cacheAt,initial,CacheInvariant,h]
  | succ k ih =>
    cases h : r.schedule <;>
      simp only [CacheInvariant,h] at ih ⊢ <;>
      simp_all [cacheAt,before,after,Nat.add_assoc]

theorem loaded_pointers (r : Request) (ba bb pid k : Nat) :
    (before r ba bb pid k (cacheAt r ba bb pid k)).ac=aAt r ba pid k ∧
    (before r ba bb pid k (cacheAt r ba bb pid k)).bc=bAt r bb pid k := by
  have h := cache_invariant r ba bb pid k
  cases hs : r.schedule <;> simp only [CacheInvariant,hs] at h <;> simp_all [before]

/-- Each k+1/k+2/k+3 source update is actual modular i64 arithmetic. -/
theorem stage_counter (k extra : Nat) : word k+word extra=word (k+extra) :=
  BitVec.ofNat_add_ofNat _ _

structure Correspondence (r : Request) (s : State Word) (a : Allocation r s) : Prop where
  metadata : ∀ n, scalarABI n=word n
  counter : ∀ pid, pid<programs r → ceilWord r.inner r.blockInner=word (count r.parameters)
  loop : ∀ pid k, pid<programs r → k≤count r.parameters →
    (word k).slt (ceilWord r.inner r.blockInner)=decide (k<count r.parameters)
  row : ∀ pid, pid<programs r → ∀ i : Fin r.blockRows,
    rowWord r pid i.val=word (row r.parameters (instanceState pid s) i)
  column : ∀ pid, pid<programs r → ∀ j : Fin r.blockColumns,
    colWord r pid j.val=word (col r.parameters (instanceState pid s) j)
  kMask : ∀ pid k, pid<programs r → k<count r.parameters → ∀ lane : Fin r.blockInner,
    loadMask r k lane.val=decide (lane.val<r.inner-k*r.blockInner)
  cMask : ∀ pid, pid<programs r → ∀ q : CIndex r,
    storeMask r pid q=active r.parameters (instanceState pid s) q
  stage : ∀ pid k,
    (before r a.baseA a.baseB pid k (cacheAt r a.baseA a.baseB pid k)).ac=aAt r a.baseA pid k ∧
    (before r a.baseA a.baseB pid k (cacheAt r a.baseA a.baseB pid k)).bc=bAt r a.baseB pid k
  pointerA : ∀ pid k, pid<programs r → ∀ i : TileIndex [r.blockRows,r.blockInner],
    aAt r a.baseA pid k i=word (a.baseA+4*aOffset r.parameters (instanceState pid s) k i)
  pointerB : ∀ pid k, pid<programs r → ∀ i : TileIndex [r.blockInner,r.blockColumns],
    bAt r a.baseB pid k i=word (a.baseB+4*bOffset r.parameters (instanceState pid s) k i)
  pointerC : ∀ pid, pid<programs r → ∀ i : CIndex r,
    cAt r a.baseC pid i=word (a.baseC+4*outputOffset r.parameters (instanceState pid s) i)
  activeA : ∀ pid k, pid<programs r → ∀ i : TileIndex [r.blockRows,r.blockInner],
    i.2.1.val<r.inner-k*r.blockInner →
      (aAt r a.baseA pid k i).toNat=a.baseA+4*aOffset r.parameters (instanceState pid s) k i
  activeB : ∀ pid k, pid<programs r → ∀ i : TileIndex [r.blockInner,r.blockColumns],
    i.1.val<r.inner-k*r.blockInner →
      (bAt r a.baseB pid k i).toNat=a.baseB+4*bOffset r.parameters (instanceState pid s) k i
  activeC : ∀ pid, pid<programs r → ∀ i : CIndex r,
    active r.parameters (instanceState pid s) i=Bool.true →
      (cAt r a.baseC pid i).toNat=a.baseC+4*outputOffset r.parameters (instanceState pid s) i

theorem correspondence (r : Request) (s : State Word) (a : Allocation r s)
    (d : Index64.Addressable r s a) (l : Index64.Launchable r) : Correspondence r s a where
  metadata := scalar_word
  counter := by
    intro pid hp
    have h := (Index64.shape_control r s a d l pid hp).2.2
    exact ceil_word _ _ r.positiveInnerBlock (by omega)
  loop := by
    intro pid k hp hk
    have h := (Index64.shape_control r s a d l pid hp).2.2
    have hc := (Index64.counts_control r s a d l pid hp).2.2
    rw [ceil_word _ _ r.positiveInnerBlock (by omega)]
    exact slt_word k (count r.parameters) (by omega) (by omega)
  row := fun pid hp => (coordinates r s a d l pid hp).1
  column := fun pid hp => (coordinates r s a d l pid hp).2
  kMask := load_mask r s a d l
  cMask := store_mask r s a d l
  stage := loaded_pointers r a.baseA a.baseB
  pointerA := a_pointer r s a d l
  pointerB := b_pointer r s a d l
  pointerC := c_pointer r s a d l
  activeA := by
    intro pid k hp i hm
    rw [a_pointer r s a d l pid k hp i]
    exact unsigned_word _ (Index64.allocated_offset _ _ _ d.input (loadA_bound r s pid k hp i hm)).2.1
  activeB := by
    intro pid k hp i hm
    rw [b_pointer r s a d l pid k hp i]
    exact unsigned_word _ (Index64.allocated_offset _ _ _ d.weight (loadB_bound r s pid k hp i hm)).2.1
  activeC := by
    intro pid hp i hm
    rw [c_pointer r s a d l pid hp i]
    exact unsigned_word _ (Index64.allocated_offset _ _ _ d.output (store_bound r s pid i hm)).2.1

#axiomsClean cache_invariant
#axiomsClean correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Native
