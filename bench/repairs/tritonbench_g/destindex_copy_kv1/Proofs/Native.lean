import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Contract

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.Native
noncomputable section
set_option maxHeartbeats 3000000

def index (c : Config) (row head dim : Nat) : BitVec 32 :=
  BitVec.ofNat 32 row * BitVec.ofNat 32 (c.heads*c.dims) +
    BitVec.ofNat 32 c.dims * BitVec.ofNat 32 head + BitVec.ofNat 32 dim

def mask (c : Config) (head dim : Nat) : Bool :=
  (BitVec.ofNat 32 head).slt (BitVec.ofNat 32 c.heads) &&
  (BitVec.ofNat 32 dim).slt (BitVec.ofNat 32 c.dims)

theorem signed32 (n : Nat) (h : n<2^31) : (BitVec.ofNat 32 n).toInt=(n:Int) :=
  BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n:Int)) (by decide) (by omega) (by norm_num; omega)

theorem index_word (c : Config) (row head dim : Nat) :
    index c row head dim=BitVec.ofNat 32 (flat c row head dim) := by
  simp [index,flat,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,Nat.mul_comm]

theorem block_bounds (m : Inputs) (w : Nat → Word) (h : InputContract m w) :
    block (config m).heads≤2^20 ∧ block (config m).dims≤2^20 := by
  have bh : 0<block (config m).heads := by unfold block; positivity
  have bd : 0<block (config m).dims := by unfold block; positivity
  have t := h.tile
  constructor <;> nlinarith

theorem mask_exact (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (a b : Nat) (ha : a<block (config m).heads) (hb : b<block (config m).dims) :
    mask (config m) a b=decide (a<(config m).heads ∧ b<(config m).dims) := by
  have blocks := block_bounds m w h
  have heads := Nat.le_pow_clog (by decide : 1<2) (config m).heads
  have dims := Nat.le_pow_clog (by decide : 1<2) (config m).dims
  change (config m).heads≤block (config m).heads at heads
  change (config m).dims≤block (config m).dims at dims
  rw [mask,BitVec.slt_eq_decide,BitVec.slt_eq_decide,signed32 a (by omega),
    signed32 b (by omega),signed32 _ (by omega),signed32 _ (by omega)]
  simp

/-- Nonnegative active source arithmetic, including all products and partial
sums, fits signed int32. Masked-off pointer values are never dereferenced. -/
theorem active_index (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (p a b : Nat) (hp : p<(config m).count) (ha : a<(config m).heads) (hb : b<(config m).dims) :
    (index (config m) p a b).toInt=(flat (config m) p a b:Int) ∧
    (index (config m) (destination w p) a b).toInt=(flat (config m) (destination w p) a b:Int) := by
  have hs := source_bound m w h p a b hp ha hb
  have ho := output_bound m w h p a b hp ha hb
  rw [index_word,index_word,signed32 _ (lt_trans hs (h.extent .source)),
    signed32 _ (lt_trans ho (h.extent .output))]
  exact ⟨rfl,rfl⟩

/-- Actual signed metadata, including its sign, agrees with the validated row. -/
theorem metadata (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (before : Memory) (binding : TensorBinding m w before) (p : Nat) (hp : p<(config m).count) :
    (before (addr (base m .destinations) p)).toInt=(destination w p:Int) := by
  rw [binding.metadata p hp]
  exact (Int.toNat_of_nonneg (h.bounds p hp).1).symm

/-- Every active signed index and scaled pointer stays in its original object. -/
theorem pointer (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (before : Memory) (binding : TensorBinding m w before) (r : Role) (x : Nat)
    (hx : x<(m r).numel) :
    (BitVec.ofNat 64 (base m r) + BitVec.ofNat 64 x * BitVec.ofNat 64 4).toNat=addr (base m r) x ∧
    addr (base m r) x+4≤base m r+4*(m r).numel ∧ addr (base m r) x+4≤2^64 := by
  have cap := binding.capacity r
  simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  have address : base m r+x*4=addr (base m r) x := by unfold addr; omega
  rw [address,Nat.mod_eq_of_lt (by unfold addr; omega)]
  unfold addr
  omega

/-- Distinct admitted output elements occupy disjoint four-byte words. -/
theorem separate_words (base i j : Nat) (h : i≠j) :
    addr base i+4≤addr base j ∨ addr base j+4≤addr base i := by unfold addr; omega

#axiomsClean mask_exact
#axiomsClean active_index
#axiomsClean metadata
#axiomsClean pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.Native
